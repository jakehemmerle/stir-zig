use ark_crypto_primitives::{
    merkle_tree::{Config, MerkleTree},
    sponge::{Absorb, CryptographicSponge},
};
use ark_ff::{FftField, PrimeField};
use ark_poly::{univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Polynomial};
use derivative::Derivative;
use std::time::Instant;

use crate::{
    ldt::Prover,
    poly_utils::{self},
    stir::{common::*, parameters::FullParameters},
    utils,
};

use crate::{domain::Domain, parameters::Parameters};

/// Enable or disable profiling output
pub static PROFILING_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn enable_profiling() {
    PROFILING_ENABLED.store(true, std::sync::atomic::Ordering::SeqCst);
}

pub fn disable_profiling() {
    PROFILING_ENABLED.store(false, std::sync::atomic::Ordering::SeqCst);
}

fn is_profiling() -> bool {
    PROFILING_ENABLED.load(std::sync::atomic::Ordering::SeqCst)
}

#[derive(Derivative)]
#[derivative(Clone(bound = "F: Clone"))]
pub struct Witness<F: FftField, MerkleConfig: Config> {
    pub(crate) domain: Domain<F>,
    pub(crate) polynomial: DensePolynomial<F>,
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,
    pub(crate) folded_evals: Vec<Vec<F>>,
}

#[derive(Derivative)]
#[derivative(Debug)]
pub struct WitnessExtended<F: FftField, MerkleConfig: Config> {
    #[derivative(Debug = "ignore")]
    pub(crate) domain: Domain<F>,
    pub(crate) polynomial: DensePolynomial<F>,

    #[derivative(Debug = "ignore")]
    pub(crate) merkle_tree: MerkleTree<MerkleConfig>,
    #[derivative(Debug = "ignore")]
    pub(crate) folded_evals: Vec<Vec<F>>,
    pub(crate) num_round: usize,
    pub(crate) folding_randomness: F,
}

pub struct StirProver<F, MerkleConfig, FSConfig>
where
    F: FftField,
    MerkleConfig: Config,
    FSConfig: CryptographicSponge,
    FSConfig::Config: Clone,
{
    pub(crate) parameters: FullParameters<F, MerkleConfig, FSConfig>,
}

impl<F, MerkleConfig, FSConfig> Prover<F, MerkleConfig, FSConfig>
    for StirProver<F, MerkleConfig, FSConfig>
where
    F: FftField + PrimeField + Absorb,
    MerkleConfig: Config<Leaf = Vec<F>>,
    MerkleConfig::InnerDigest: Absorb,
    FSConfig: CryptographicSponge,
    FSConfig::Config: Clone,
{
    type FullParameter = FullParameters<F, MerkleConfig, FSConfig>;
    type Commitment = Commitment<MerkleConfig>;
    type Witness = Witness<F, MerkleConfig>;
    type Proof = Proof<F, MerkleConfig>;

    fn new(parameters: Parameters<F, MerkleConfig, FSConfig>) -> Self {
        Self::new_full(parameters.into())
    }

    fn new_full(full_parameters: Self::FullParameter) -> Self {
        Self {
            parameters: full_parameters,
        }
    }

    // TODO: Rename witness_polynomial
    fn commit(
        &self,
        witness_polynomial: DensePolynomial<F>,
    ) -> (Commitment<MerkleConfig>, Witness<F, MerkleConfig>) {
        let domain = Domain::<F>::new(
            self.parameters.starting_degree,
            self.parameters.starting_rate,
        )
        .unwrap();

        let fft_start = Instant::now();
        let evals = witness_polynomial
            .evaluate_over_domain_by_ref(domain.backing_domain)
            .evals;
        let folded_evals = utils::stack_evaluations(evals, self.parameters.folding_factor);
        let fft_time = fft_start.elapsed();

        let merkle_start = Instant::now();
        let merkle_tree = MerkleTree::<MerkleConfig>::new(
            &self.parameters.leaf_hash_params,
            &self.parameters.two_to_one_params,
            &folded_evals,
        )
        .unwrap();
        let merkle_time = merkle_start.elapsed();

        if is_profiling() {
            println!("      [commit] FFT: {:.2}ms, Merkle: {:.2}ms",
                fft_time.as_secs_f64() * 1000.0,
                merkle_time.as_secs_f64() * 1000.0);
        }

        let initial_commitment = merkle_tree.root();

        (
            Commitment {
                root: initial_commitment,
            },
            Witness {
                domain,
                polynomial: witness_polynomial,
                merkle_tree,
                folded_evals,
            },
        )
    }

    fn prove(&self, witness: Witness<F, MerkleConfig>) -> Proof<F, MerkleConfig> {
        assert!(witness.polynomial.degree() < self.parameters.starting_degree);

        let mut sponge = FSConfig::new(&self.parameters.fiat_shamir_config);
        // TODO: Add parameters to FS
        sponge.absorb(&witness.merkle_tree.root());
        let folding_randomness = sponge.squeeze_field_elements(1)[0];

        let mut witness = WitnessExtended {
            domain: witness.domain,
            polynomial: witness.polynomial,
            merkle_tree: witness.merkle_tree,
            folded_evals: witness.folded_evals,
            num_round: 0,
            folding_randomness,
        };

        let mut round_proofs = vec![];
        for round_num in 0..self.parameters.num_rounds {
            let round_start = Instant::now();
            let (new_witness, round_proof) = self.round(&mut sponge, &witness);
            let round_time = round_start.elapsed();
            if is_profiling() {
                println!("      [prove] Round {}: {:.2}ms", round_num, round_time.as_secs_f64() * 1000.0);
            }
            witness = new_witness;
            round_proofs.push(round_proof);
        }

        let final_fold_start = Instant::now();
        let final_polynomial = poly_utils::folding::poly_fold(
            &witness.polynomial,
            self.parameters.folding_factor,
            witness.folding_randomness,
        );

        let final_repetitions = self.parameters.repetitions[self.parameters.num_rounds];
        let scaling_factor = witness.domain.size() / self.parameters.folding_factor;
        let final_randomness_indexes = utils::dedup(
            (0..final_repetitions).map(|_| utils::squeeze_integer(&mut sponge, scaling_factor)),
        );

        let queries_to_final_ans: Vec<_> = final_randomness_indexes
            .iter()
            .map(|index| witness.folded_evals[*index].clone())
            .collect();

        let queries_to_final_proof = witness
            .merkle_tree
            .generate_multi_proof(final_randomness_indexes)
            .unwrap();

        let queries_to_final = (queries_to_final_ans, queries_to_final_proof);
        let final_fold_time = final_fold_start.elapsed();

        let pow_start = Instant::now();
        let pow_nonce = utils::proof_of_work(
            &mut sponge,
            self.parameters.pow_bits[self.parameters.num_rounds],
        );
        let pow_time = pow_start.elapsed();

        if is_profiling() {
            println!("      [prove] Final fold: {:.2}ms, PoW: {:.2}ms",
                final_fold_time.as_secs_f64() * 1000.0,
                pow_time.as_secs_f64() * 1000.0);
        }

        Proof {
            round_proofs,
            final_polynomial,
            queries_to_final,
            pow_nonce,
        }
    }
}

impl<F, MerkleConfig, FSConfig> StirProver<F, MerkleConfig, FSConfig>
where
    F: FftField + PrimeField + Absorb,
    MerkleConfig: Config<Leaf = Vec<F>>,
    MerkleConfig::InnerDigest: Absorb,
    FSConfig: CryptographicSponge,
    FSConfig::Config: Clone,
{
    pub fn new(parameters: Parameters<F, MerkleConfig, FSConfig>) -> Self {
        Self {
            parameters: parameters.into(),
        }
    }

    // TODO: Rename to better name
    fn round(
        &self,
        sponge: &mut impl CryptographicSponge,
        witness: &WitnessExtended<F, MerkleConfig>,
    ) -> (
        WitnessExtended<F, MerkleConfig>,
        RoundProof<F, MerkleConfig>,
    ) {
        let round_start = Instant::now();

        let fold_start = Instant::now();
        let g_poly = poly_utils::folding::poly_fold(
            &witness.polynomial,
            self.parameters.folding_factor,
            witness.folding_randomness,
        );
        let fold_time = fold_start.elapsed();

        // TODO: For now, I am FFTing
        let fft_start = Instant::now();
        let g_domain = witness.domain.scale_offset(2);
        let g_evaluations = g_poly
            .evaluate_over_domain_by_ref(g_domain.backing_domain)
            .evals;
        let g_folded_evaluations =
            utils::stack_evaluations(g_evaluations, self.parameters.folding_factor);
        let fft_time = fft_start.elapsed();

        let merkle_start = Instant::now();
        let g_merkle = MerkleTree::<MerkleConfig>::new(
            &self.parameters.leaf_hash_params,
            &self.parameters.two_to_one_params,
            &g_folded_evaluations,
        )
        .unwrap();
        let g_root = g_merkle.root();
        let merkle_time = merkle_start.elapsed();

        sponge.absorb(&g_root);

        // Out of domain sample
        let ood_randomness = sponge.squeeze_field_elements(self.parameters.ood_samples);
        let betas: Vec<F> = ood_randomness
            .iter()
            .map(|alpha| g_poly.evaluate(alpha))
            .collect();
        sponge.absorb(&betas);

        // Proximity generator
        let comb_randomness: F = sponge.squeeze_field_elements(1)[0];

        // Folding randomness for next round
        let folding_randomness = sponge.squeeze_field_elements(1)[0];

        // Sample the indexes of L^k that we are going to use for querying the previous Merkle tree
        let scaling_factor = witness.domain.size() / self.parameters.folding_factor;
        let num_repetitions = self.parameters.repetitions[witness.num_round];
        let stir_randomness_indexes = utils::dedup(
            (0..num_repetitions).map(|_| utils::squeeze_integer(sponge, scaling_factor)),
        );

        let pow_start = Instant::now();
        let pow_nonce = utils::proof_of_work(sponge, self.parameters.pow_bits[witness.num_round]);
        let pow_time = pow_start.elapsed();

        // Not used
        let _shake_randomness: F = sponge.squeeze_field_elements(1)[0];

        // The verifier queries the previous oracle at the indexes of L^k (reading the
        // corresponding evals)
        let queries_to_prev_ans: Vec<_> = stir_randomness_indexes
            .iter()
            .map(|&index| witness.folded_evals[index].clone())
            .collect();

        let queries_to_prev_proof = witness
            .merkle_tree
            .generate_multi_proof(stir_randomness_indexes.clone())
            .unwrap();
        let queries_to_prev = (queries_to_prev_ans, queries_to_prev_proof);

        // Here, we update the witness
        // First, compute the set of points we are actually going to query at
        let stir_randomness: Vec<_> = stir_randomness_indexes
            .iter()
            .map(|index| {
                witness
                    .domain
                    .scale(self.parameters.folding_factor)
                    .element(*index)
            })
            .collect();


        let beta_answers = betas
            .iter()
            .zip(ood_randomness.iter())
            .map(|(y, x)| (*x, *y))
            .collect::<Vec<_>>();

        let quotient_answers = stir_randomness
            .iter()
            .map(|x| (*x, g_poly.evaluate(x)))
            .chain(beta_answers.into_iter())
            .collect::<Vec<_>>();

        // Then compute the set we are quotienting by
        let quotient_set: Vec<_> = ood_randomness
            .into_iter()
            .chain(stir_randomness.iter().cloned())
            .collect();

        let interp_start = Instant::now();
        let ans_polynomial = poly_utils::interpolation::naive_interpolation(&quotient_answers);

        let mut shake_polynomial = DensePolynomial::from_coefficients_vec(vec![]);
        for (x, y) in quotient_answers {
            let num_polynomial = &ans_polynomial - &DensePolynomial::from_coefficients_vec(vec![y]);
            let den_polynomial = DensePolynomial::from_coefficients_vec(vec![-x, F::ONE]);
            shake_polynomial = shake_polynomial + (&num_polynomial / &den_polynomial);
        }
        let interp_time = interp_start.elapsed();

        // The quotient_polynomial is then computed
        let div_start = Instant::now();
        let vanishing_poly = poly_utils::interpolation::vanishing_poly(&quotient_set);
        // Resue the ans_polynomial to compute the quotient_polynomial
        let numerator = &g_poly + &ans_polynomial;
        let quotient_polynomial = &numerator / &vanishing_poly;

        // This is the polynomial 1 + r * x + r^2 * x^2 + ... + r^n * x^n where n = |quotient_set|
        let scaling_polynomial = DensePolynomial::from_coefficients_vec(
            (0..quotient_set.len() + 1)
                .map(|i| comb_randomness.pow([i as u64]))
                .collect(),
        );

        let witness_polynomial = &quotient_polynomial * &scaling_polynomial;
        let div_time = div_start.elapsed();

        let round_total = round_start.elapsed();
        if is_profiling() {
            println!("        [round] fold:{:.1}ms fft:{:.1}ms merkle:{:.1}ms pow:{:.1}ms interp:{:.1}ms div:{:.1}ms total:{:.1}ms",
                fold_time.as_secs_f64() * 1000.0,
                fft_time.as_secs_f64() * 1000.0,
                merkle_time.as_secs_f64() * 1000.0,
                pow_time.as_secs_f64() * 1000.0,
                interp_time.as_secs_f64() * 1000.0,
                div_time.as_secs_f64() * 1000.0,
                round_total.as_secs_f64() * 1000.0);
        }

        (
            WitnessExtended {
                domain: g_domain,
                polynomial: witness_polynomial,
                merkle_tree: g_merkle,
                folded_evals: g_folded_evaluations,
                num_round: witness.num_round + 1,
                folding_randomness,
            },
            RoundProof {
                g_root,
                betas,
                queries_to_prev,
                ans_polynomial,
                shake_polynomial,
                pow_nonce,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::fields::Field64 as TestField;
    use ark_ff::Field;

    fn fold_evaluations<F: FftField>(folding: usize, evals: Vec<F>) -> Vec<Vec<F>> {
        let size_of_new_domain = evals.len() / folding;

        let mut folded_evals = vec![];
        for i in 0..size_of_new_domain {
            let mut new_evals = vec![];
            for j in 0..folding {
                new_evals.push(evals[i + j * size_of_new_domain]);
            }
            folded_evals.push(new_evals);
        }

        folded_evals
    }

    #[test]
    fn test_folding() {
        let folding = 4;
        let domain = Domain::<TestField>::new(16, 4).unwrap();

        let elements = domain.elements().collect::<Vec<_>>();
        let root = elements[1];

        let folded_evals = fold_evaluations(folding, elements.clone());

        let new_size = elements.len() / folding;

        for i in 0..new_size {
            let index_element = root.pow([(i * folding) as u64]);
            for j in 0..folding {
                assert_eq!(folded_evals[i][j].pow([folding as u64]), index_element);
            }
        }
    }
}
