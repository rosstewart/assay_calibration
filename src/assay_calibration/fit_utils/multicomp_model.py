from typing import List, Optional, Dict
from sklearn.cluster import KMeans
import numpy as np
from scipy.stats import skewnorm, norm
from scipy.special import logsumexp
from copy import deepcopy
from tqdm import tqdm
from optimization_init import optimize_distributions


class MulticomponentCalibrationModel:
    """
    Multi-component skew-normal calibration model.

    Represent functionally normal (FN) and functionally abnormal (FA) distributions each as mixtures of skew normal distributions.

    Monotonicity is enforced between each pair of neighboring FN and FA components.

    Pathogenic, benign, and gnomAD assay-score distributions are modeled as mixtures of FN and FA mixture-distributions.
    """

    @classmethod
    def from_params(cls, skewness, locs, scales, sample_weights, **kwargs):
        """
        Initialize the model from the given parameters.

        Parameters
        ----------
        - skewness : numpy.array
            Skewness parameters for each component.
        - locs : numpy.array
            Location parameters for each component.
        - scales : numpy.array
            Scale parameters for each component.
        - sample_weights : numpy.array
            Sample weights for each component.

        """
        model = cls(len(skewness))
        model.skewness = skewness
        model.locs = locs
        model.scales = scales
        model.sample_weights = np.array(sample_weights)
        return model

    def __init__(self, num_components, **kwargs):
        """
        Initialize the model with the given component classes.

        Parameters
        ----------
        num_components : int
            Number of components in the model.
        """
        self.num_components = num_components

    def fit(self, scores, sampleIndicators, **kwargs):
        """
        Fit the model to the given assay scores and sample indicators.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]

        Optional Parameters
        -------------------
        - check_convergence : bool (default True)
            If True, check for convergence in the log likelihood
        - verbose : bool (default False)
            If True, print progress messages.
        - max_iter : int (default 10,000)
            Maximum number of iterations to run the EM algorithm.
        - tol : float (default 1e-6)
            Tolerance for convergence in the log likelihood.
        - check_monotonic : bool (default True)
            If True, check for monotonicity between each pair of neighboring components
        - score_min : float | int (default None)
            Minimum score to consider when checking for monotonicity. If None, use the minimum score in `scores`.
        - score_max : float | int (default None)
            Maximum score to consider when checking for monotonicity. If None, use the maximum score in `scores`.
        """
        self._max_likelihood = -np.inf
        self.check_convergence = kwargs.pop("check_convergence", True)
        self.check_monotonic = kwargs.pop("check_monotonic", True)
        self.score_min = kwargs.get("score_min", None)
        self.score_max = kwargs.get("score_max", None)
        sampleIndicators = sampleIndicators.astype(bool)
        # Validate input data
        self.validate_inputs(scores, sampleIndicators)
        # Initialize model parameters (i.e., skewness, locs, scales, sample_weights)
        self.initialize_parameters(scores, sampleIndicators, **kwargs)
        # run the EM algorithm to fit the model to the given assay scores and sample indicators
        self._max_iter = kwargs.get("max_iter", 10000)
        self._tol = kwargs.get("tol", 1e-6)
        self._iter = 0
        self._iters_since_improvement = 0
        self._log_likelihoods = []
        self._update_log_likelihood(scores, sampleIndicators)
        pbar = None
        if kwargs.get("verbose", False):
            pbar = tqdm(total=self._max_iter)
        while not self.has_converged(**kwargs):
            if self.check_monotonic and self.any_components_violate_monotonicity(
                scores
            ):
                raise ValueError(
                    f"Model parameters violate monotonicity at start of iteration {self._iter:,d}."
                )
            if pbar is not None:
                pbar.update(1)
                pbar.set_description(f"Log-likelihood: {self._log_likelihoods[-1]:.7f}")
            if self.check_monotonic and self.any_components_violate_monotonicity(
                scores
            ):
                raise ValueError(
                    f"Model parameters violate monotonicity at start of iteration {self._iter:,d}."
                )
            self._fit_iter(scores, sampleIndicators, **kwargs)
            self._iter += 1
            self._update_log_likelihood(scores, sampleIndicators)
        if pbar is not None:
            pbar.close()

    def has_converged(self, **kwargs):
        """
        Check if the model has converged.

        Kwargs:
        - patience : int (default 25)
            Number of iterations to wait for improvement in the log likelihood before stopping.
        - verbose : bool (default False)
            If True, print progress messages.

        Returns
        -------
        bool
        """
        patience = kwargs.get("patience", 25)
        verbose = kwargs.get("verbose", False)
        if self._log_likelihoods[-1] > self._max_likelihood:
            self._iters_since_improvement = 0
            self._max_likelihood = self._log_likelihoods[-1]
            return False
        self._iters_since_improvement += 1
        if self._iter >= self._max_iter:
            if verbose:
                print(f"Reached maximum iterations ({self._max_iter}).")
            return True
        if np.isinf(self._log_likelihoods[-1]):
            if verbose:
                print("Log likelihood is infinite.")
            return True
        if self.check_convergence and self._iters_since_improvement > patience:
            if verbose:
                print(f"No improvement in log likelihood for {patience} iterations.")
            return True
        return False

    def _update_log_likelihood(self, scores, sampleIndicators):
        """
        Update the likelihood of the model.
        """
        self._log_likelihoods.append(self.get_log_likelihood(scores, sampleIndicators))

    def _fit_iter(self, scores, sampleIndicators, **kwargs):
        """
        Run the EM algorithm for a single iteration.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.

        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]

        Returns
        -------
        None
        """

        # step 1) make sure monotonicity is enforced
        if self.check_monotonic and self.any_components_violate_monotonicity(
            sorted(np.unique(scores))
        ):
            raise ValueError(
                f"Model parameters violate monotonicity at iteration {self._iter:,d}."
            )
        # step 2) update the parameters of each component
        component_posteriors = self.get_component_posteriors(scores, sampleIndicators)
        for component_num in range(self.num_components):
            self._current_component = component_num
            if self.check_monotonic and self.any_components_violate_monotonicity(
                sorted(np.unique(scores))
            ):
                raise ValueError(
                    f"Model parameters violate monotonicity at start of component {component_num} iteration {self._iter:,d}."
                )
            self._update_component_parameters(
                scores, component_posteriors[:, component_num], component_num
            )
            if self.check_monotonic and self.any_components_violate_monotonicity(
                sorted(np.unique(scores))
            ):
                raise ValueError(
                    f"Updated component parameters for component {component_num} at iteration {self._iter} violate monotonicity.\n{self.get_params()}"
                )
        # step 3) update mixture weights for each sample
        self._update_sample_weights(scores, sampleIndicators)
        if self.check_monotonic and self.any_components_violate_monotonicity(
            sorted(np.unique(scores))
        ):
            raise ValueError(
                f"Model parameters violate monotonicity at end of iteration {self._iter:,d}."
            )

    def _update_component_parameters(
        self, scores, component_posteriors, component_num, **kwargs
    ) -> None:
        """
        Update the parameters of the given component.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        component_posteriors : numpy.array
            Posterior probabilities of each sample being from the given component.
        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]
        component_num : int
            Component number.

        Returns
        -------
        None
        """
        self._update_component_location(
            scores, component_posteriors, component_num, **kwargs
        )
        self._update_component_Delta(
            scores, component_posteriors, component_num, **kwargs
        )
        self._update_component_Gamma(
            scores, component_posteriors, component_num, **kwargs
        )
        (
            self.skewness[component_num],
            self.locs[component_num],
            self.scales[component_num],
        ) = self.alternate_to_canonical(
            self.updated_component_location,
            self.updated_component_Delta,
            self.updated_component_Gamma,
        )
        # make sure the updated component parameters satisfy monotonicity
        if self.check_monotonic and self.any_components_violate_monotonicity(
            sorted(np.unique(scores))
        ):
            raise ValueError(
                f"Updated component parameters for component {component_num} at iteration {self._iter} violate monotonicity.\n{self.get_params()}"
            )

    def _update_sample_weights(self, scores, sampleIndicators, **kwargs) -> None:
        """
        Update the sample weights.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]

        Returns
        -------
        None
        """
        component_posteriors = self.get_component_posteriors(scores, sampleIndicators)
        for sampleIdx in range(sampleIndicators.shape[1]):
            sample_component_posteriors = component_posteriors[
                sampleIndicators[:, sampleIdx]
            ]
            self.sample_weights[sampleIdx] = np.mean(
                sample_component_posteriors, axis=0
            )

    def get_component_params(self, component_num):
        """
        Get the parameters of the given component.

        Parameters
        ----------
        component_num : int
            Component number.

        Returns
        -------
        List[float]
            Component parameters (skewness, loc, scale).
        """
        return (
            self.skewness[component_num],
            self.locs[component_num],
            self.scales[component_num],
        )

    def _update_component_location(
        self, scores, component_posteriors, component_num, **kwargs
    ) -> None:
        """
        Update the location parameter of the given component.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.

        component_posteriors : numpy.array
            Posterior probabilities of each sample being from the given component.

        component_num : int
            Component number.

        Returns
        -------
        None
        """
        parameter_idx = (
            0  # index of the location parameter within the alternate parameterization
        )
        location_candidate = self._propose_location_update(
            scores, component_posteriors, self.get_component_params(component_num)
        )
        if not self.check_monotonic:
            updated_location = location_candidate
            self.updated_component_location = updated_location
            return
        # get the last pair of component parameters for components component_num and component_num + 1 that satistfied monotonicity
        last_params_canonical = [
            self.get_component_params(component_num)
            for component_num in range(self.num_components)
        ]
        # assign the index of the component within the tuple to update
        updated_location = self.binary_search(
            scores,
            location_candidate,
            last_params_canonical,
            parameter_idx,
            component_num,
        )

        self.updated_component_location = updated_location

    def _propose_location_update(self, scores, component_posteriors, component_params):
        """
        Propose a new location parameter for the given component, ignoring density constraint.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        component_posteriors : numpy.array
            Posterior probabilities of each sample being from the given component.
        component_params : List[float]
            Component parameters (skewness, loc, scale).
        component_num : int
            Component number.

        Returns
        -------
        float
            Proposed location parameter.
        """
        v, _ = self.get_truncated_normal_moments(scores, component_params)
        (_, Delta, _) = MulticomponentCalibrationModel.canonical_to_alternate(
            *component_params
        )
        m = scores - v * Delta
        candidate = (m * component_posteriors).sum() / component_posteriors.sum()
        return candidate

    @staticmethod
    def canonical_to_alternate(skewness, location, scale):
        """
        convert canonical parameters to alternate parameters

        Arguments:
        a: skewness parameter
        location: location parameter
        scale: scale parameter

        Returns:
        - location
        - Delta
        - Gamma
        """
        Delta = 0
        Gamma = 0

        _delta = skewness / np.sqrt(1 + skewness**2)
        Delta = scale * _delta
        Gamma = scale**2 - Delta**2

        return tuple(map(float, (location, Delta, Gamma)))

    @staticmethod
    def alternate_to_canonical(loc, Delta, Gamma):
        """
        convert alternate parameters to canonical parameters

        Arguments:
        - loc: location parameter
        - Delta: Delta parameter
        - Gamma: Gamma parameter

        Returns:
        skewness: skewness parameter
        location: location parameter
        scale: scale parameter
        """
        try:
            skewness = np.sign(Delta) * np.sqrt(Delta**2 / Gamma)
        except ZeroDivisionError:
            raise ZeroDivisionError(
                f"Invalid skewness parameter: {np.sign(Delta) * np.sqrt(Delta**2 / Gamma)} from Delta: {Delta}, Gamma: {Gamma}"
            )
        if np.isinf(skewness) or np.isnan(skewness):
            raise ZeroDivisionError(
                f"Invalid skewness parameter: {skewness} from Delta: {Delta}, Gamma: {Gamma}"
            )
        scale = np.sqrt(Gamma + Delta**2)
        return tuple(map(float, (skewness, loc, scale)))

    def _update_component_Delta(
        self, scores, component_posteriors, component_num, **kwargs
    ) -> None:
        """
        Update the Delta parameter of the given component.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        component_posteriors : numpy.array
            Posterior probabilities of each sample being from the given component.
        component_num : int
            Component number.

        Returns
        -------
        None
        """
        parameter_idx = (
            1  # index of the Delta parameter within the alternate parameterization
        )
        Delta_candidate = self._propose_Delta_update(
            scores, component_posteriors, component_num
        )
        if not self.check_monotonic:
            updated_Delta = Delta_candidate
            self.updated_component_Delta = updated_Delta
            return
        last_params_canonical = []
        for i in range(self.num_components):
            if i != component_num:
                last_params_canonical.append(self.get_component_params(i))
            else:
                (_, D_i, G_i) = MulticomponentCalibrationModel.canonical_to_alternate(
                    *self.get_component_params(i)
                )
                (skewness_i, loc_i, scale_i) = self.alternate_to_canonical(
                    self.updated_component_location, D_i, G_i
                )
                last_params_canonical.append((skewness_i, loc_i, scale_i))
        updated_Delta = self.binary_search(
            scores, Delta_candidate, last_params_canonical, parameter_idx, component_num
        )
        self.updated_component_Delta = updated_Delta

    def _propose_Delta_update(self, scores, component_posteriors, component_num):
        """
        Propose a new Delta parameter for the given component, ignoring density constraint.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        component_posteriors : numpy.array
            Posterior probabilities of each sample being from the given component.
        component_num : int
            Component number.

        Returns
        -------
        float
            Proposed Delta parameter.
        """
        skewness_i, loc_i, scale_i = self.get_component_params(component_num)
        v, _ = self.get_truncated_normal_moments(scores, (skewness_i, loc_i, scale_i))
        d = v * (scores - loc_i)
        candidate = (d * component_posteriors).sum() / component_posteriors.sum()
        return candidate

    def _update_component_Gamma(
        self, scores, component_posteriors, component_num, **kwargs
    ) -> None:
        """
        Update the Gamma parameter of the given component.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        component_posteriors : numpy.array
            Posterior probabilities of each sample being from the given component.
        component_num : int
            Component number.

        Returns
        -------
        None
        """
        parameter_idx = (
            2  # index of the Gamma parameter within the alternate parameterization
        )
        Gamma_candidate = self._propose_Gamma_update(
            scores, component_posteriors, component_num
        )
        if not self.check_monotonic:
            updated_Gamma = Gamma_candidate
            self.updated_component_Gamma = updated_Gamma
            return
        last_params_canonical = []
        for i in range(self.num_components):
            if i != component_num:
                last_params_canonical.append(self.get_component_params(i))
            else:
                (_, _, G_i) = MulticomponentCalibrationModel.canonical_to_alternate(
                    *self.get_component_params(i)
                )
                (skewness_i, loc_i, scale_i) = self.alternate_to_canonical(
                    self.updated_component_location, self.updated_component_Delta, G_i
                )
                last_params_canonical.append((skewness_i, loc_i, scale_i))
        updated_Gamma = self.binary_search(
            scores, Gamma_candidate, last_params_canonical, parameter_idx, component_num
        )
        self.updated_component_Gamma = updated_Gamma

    def _propose_Gamma_update(self, scores, component_posteriors, component_num):
        """
        Propose a new Gamma parameter for the given component, ignoring density constraint.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        component_posteriors : numpy.array
            Posterior probabilities of each sample being from the given component.
        component_num : int
            Component number.

        Returns
        -------
        float
            Proposed Gamma parameter.
        """
        skewness_i, loc_i, scale_i = self.get_component_params(component_num)
        v, w = MulticomponentCalibrationModel.get_truncated_normal_moments(
            scores, (skewness_i, loc_i, scale_i)
        )
        g = (
            (scores - self.updated_component_location) ** 2
            - (
                2
                * self.updated_component_Delta
                * v
                * (scores - self.updated_component_location)
            )
            + (self.updated_component_Delta**2 * w)
        )
        return (g * component_posteriors).sum() / component_posteriors.sum()

    @staticmethod
    def get_truncated_normal_moments(observations, component_params):
        """
        Get the first and second moments of the truncated normal distribution.
        """
        _delta = MulticomponentCalibrationModel._get_delta(component_params[0])
        loc, scale = component_params[1:]
        truncated_normal_loc = _delta / scale * (observations - loc)
        truncated_normal_scale = np.sqrt(1 - _delta**2)
        v, w = MulticomponentCalibrationModel.trunc_norm_moments(
            truncated_normal_loc, truncated_normal_scale
        )
        return v, w

    @staticmethod
    def trunc_norm_moments(mu, sigma):
        """first and second truncated normal moments"""
        cdf = norm.cdf(mu / sigma)
        flags = cdf == 0
        pdf = norm.pdf(mu / sigma)
        p = np.zeros_like(pdf)
        p[~flags] = pdf[~flags] / cdf[~flags]
        p[flags] = abs(mu[flags] / sigma)
        m1 = mu + sigma * p
        m2 = mu**2 + sigma**2 + sigma * mu * p
        return m1, m2

    @staticmethod
    def _get_delta(skewness):
        """utility value for parameter updates"""
        return skewness / np.sqrt(1 + skewness**2)

    def validate_inputs(self, scores, sampleIndicators):
        """
        Validate the input data.
        """
        nscores = scores.shape[0]
        nindicators, nsamples = sampleIndicators.shape
        assert (
            nscores == nindicators
        ), f"The number of scores ({nscores})must match the number of samples ({nindicators})."
        assert np.all(
            np.sum(sampleIndicators, axis=1) == 1
        ), "sampleIndicators is expected to be a one-hot matrix."
        assert np.all(
            np.sum(sampleIndicators, axis=0) > 0
        ), "each sample must have at least one observation."

    def initialize_parameters(
        self,
        scores,
        sampleIndicators,
        skew_directions: Optional[List[int]] = None,
        max_skew_init_magnitude=1,
        **kwargs,
    ) -> None:
        """
        Initialize the model parameters.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]

        Optional Parameters
        -------------------
        - skew-directions : List[int] | None
            List of skew directions for each component, where each skew direction is either 1 (right-skewed), 0 (standard-normal), or -1 (left-skewed).
            If None, randomly assign skew directions to each component.
        - max-skew-init-magnitude : float | int (default 1)
            Maximum magnitude of the skew parameter for each skew-normal component.

        Returns
        -------
        None
        """
        if kwargs.get("verbose", False):
            print("Initializing model parameters...")
        initializations = 0
        initialized = False
        while not initialized and initializations < 100:
            if kwargs.get("verbose", False):
                print(f"Initialization {initializations}...")
            # 1) Fit a k-means model to all assay scores
            self.kmeans_model = KMeans(
                n_clusters=self.num_components, init="random", n_init=1
            )
            scores = scores.reshape((-1, 1))
            self.kmeans_model.fit(
                scores[np.random.randint(0, scores.shape[0], scores.shape[0])]
            )
            # reorder cluster_centers_ from min to max
            self.kmeans_model.cluster_centers_ = np.sort(
                self.kmeans_model.cluster_centers_.ravel()
            )[..., None]
            component_assignments = self.kmeans_model.predict(scores)
            comp_nums, comp_counts = np.unique(
                component_assignments, return_counts=True
            )
            if len(comp_nums) < self.num_components or (comp_counts < 2).any():
                if kwargs.get("verbose", False):
                    print(
                        f"Identified {len(comp_nums)} components with counts {comp_counts}."
                    )
                initializations += 1
                continue
            # 2) Initialize skew-normal component parameters
            self._initialize_skew_normal_parameters(
                scores,
                component_assignments,
                skew_directions,
                max_skew_init_magnitude,
                **kwargs,
            )
            if (self.scales == 0).any():
                if kwargs.get("verbose", False):
                    print(f"Found a component with scale 0.\n{self.scales}")
                initializations += 1
                continue
            # 3) Initialize the mixture weights
            self._initialize_sample_weights(scores, sampleIndicators)

            # 4) adjust the skew-normal component parameters to enforce monotonicity between adjacent components
            if not self.check_monotonic:
                initialized = True
                break
            initialized = self.adjust_to_monotonicity(np.unique(scores))
            initializations += 1
        if not initialized:
            raise ValueError(
                f"Could not initialize model parameters; Last params:\nskewness:{self.skewness}\nlocs:{self.locs}\nscales:{self.scales}"
            )
        if self.check_monotonic and self.any_components_violate_monotonicity(
            np.unique(scores)
        ):
            raise ValueError("Model parameters violate monotonicity.")
        if kwargs.get("verbose", False):
            print("Model parameters initialized.")
            print(f"Skews: {self.skewness}\nLocs: {self.locs}\nScales: {self.scales}")

    def _initialize_skew_normal_parameters(
        self,
        scores: np.ndarray,
        component_assignments: np.ndarray,
        skew_directions: Optional[List[int]],
        max_skew_init_magnitude: float | int,
        **kwargs,
    ) -> None:
        """
        Initialize the skew-normal component parameters.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - component_assignments : numpy.array
            Component assignments (from k-means) for each assay score.
        - skew-directions : List[int] | None
            List of skew directions for each component, where each skew direction is either 1 (right-skewed), 0 (standard-normal), or -1 (left-skewed).
            If None, randomly assign skew directions to each component.
        - max-skew-init-magnitude : float | int
            Maximum magnitude of the skew parameter for each skew-normal component.

        Returns
        -------
        None
        """
        # initialize the skew-normal component parameters
        if skew_directions is None:
            skew_directions = np.random.choice([-1, 0, 1], self.num_components)
        else:
            assert (
                len(skew_directions) == self.num_components
            ), "The number of skew directions must match the number of components."
            assert all(
                skew_direction in [-1, 0, 1] for skew_direction in skew_directions
            ), "Skew directions must be either -1 (left-skewed), 0 (standard-normal), or 1 (right-skewed)."
        assert skew_directions is not None, "skew_directions must be provided."
        self.skewness = np.zeros(self.num_components, dtype=float)
        self.locs = np.zeros(self.num_components, dtype=float)
        self.scales = np.zeros(self.num_components, dtype=float)
        indexMapping = self._groupValues(component_assignments)
        for componentNum, skewDirection in enumerate(skew_directions):
            if skewDirection == 1:
                self.skewness[componentNum] = np.random.uniform(
                    0, max_skew_init_magnitude
                )
            elif skewDirection == -1:
                self.skewness[componentNum] = np.random.uniform(
                    -max_skew_init_magnitude, 0
                )
            component_scores = scores[indexMapping[componentNum]]
            self.locs[componentNum] = np.mean(component_scores)
            self.scales[componentNum] = np.std(component_scores)

    def adjust_to_monotonicity(self, scores: np.ndarray, **kwargs) -> bool:
        """
        Adjust the skew-normal component parameters to enforce monotonicity between the FN and FA components.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.

        Optional Parameters
        -------------------
        - max_monotonicity_reduction_iters : int (default 100)
            Maximum number of iterations to reduce the skewness and scales parameters to enforce monotonicity.

        Returns
        -------
        - initialized : bool
            True if the skew-normal component parameters were successfully adjusted to enforce monotonicity, False otherwise
        """
        initialParmas = [
            [self.skewness[i], self.locs[i], self.scales[i]]
            for i in range(self.num_components)
        ]
        smin = scores.min()
        smax = scores.max()
        if self.score_min is not None:
            smin = self.score_min
        if self.score_max is not None:
            smax = self.score_max
        score_range = (smin, smax)
        optimized_params = optimize_distributions(
            [tuple(param) for param in initialParmas], x_range=score_range
        )
        self.skewness = np.array([params[0] for params in optimized_params])
        self.locs = np.array([params[1] for params in optimized_params])
        self.scales = np.array([params[2] for params in optimized_params])
        if self.any_components_violate_monotonicity(scores, **kwargs):
            if kwargs.get("verbose", False):
                print("density constraint initialization failed.")
            return False
        return True

    def any_components_violate_monotonicity(self, scores, **kwargs) -> bool:
        """
        Check whether the joint density ratio of all adjacent skew normal components is monotonic

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.

        Optional Parameters
        -------------------
        - score_min : float | int (default None)
            Minimum score to consider when checking for monotonicity.
        - score_max : float | int (default None)
            Maximum score to consider when checking for monotonicity.

        Returns
        -------
        bool
            True if the joint density ratio of any pair of adjacent skew normal components is non-monotonic, False otherwise.
        """
        uscores = np.unique(scores)
        uscores.sort()
        if self.score_min is not None:
            if self.score_min < uscores[0]:
                uscores = np.insert(uscores, 0, self.score_min)
            uscores = uscores[uscores >= self.score_min]
        if self.score_max is not None:
            if self.score_max > uscores[-1]:
                uscores = np.append(
                    uscores,
                    [
                        self.score_max,
                    ],
                )
            uscores = uscores[uscores <= self.score_max]
        for comp_i in range(self.num_components):
            if comp_i != self.num_components - 1:
                # if comp_i is not the last component, check that the density ratio of comp_i to comp_i + 1 is monotonic
                comp_j = comp_i + 1
                if MulticomponentCalibrationModel.parameters_violate_monotonicity(
                    [
                        self.skewness[comp_i],
                        self.locs[comp_i],
                        self.scales[comp_i],
                    ],
                    [
                        self.skewness[comp_j],
                        self.locs[comp_j],
                        self.scales[comp_j],
                    ],
                    self.score_min,
                    self.score_max,
                ):
                    return True
            if comp_i != 0:
                # if comp_i is not the first component, check that the density ratio of comp_i to comp_i - 1 is monotonic
                comp_j = comp_i - 1
                if MulticomponentCalibrationModel.parameters_violate_monotonicity(
                    [
                        self.skewness[comp_j],
                        self.locs[comp_j],
                        self.scales[comp_j],
                    ],
                    [
                        self.skewness[comp_i],
                        self.locs[comp_i],
                        self.scales[comp_i],
                    ],
                    self.score_min,
                    self.score_max,):
                        return True
        return False

    @staticmethod
    def parameters_violate_monotonicity(params_i, params_j,score_min,score_max) -> bool:
        """
        Check whether the density ratio of component i to component j is monotonic.

        Parameters
        ----------
        - params_i : List[float]
            Component parameters for component i (skewness, loc, scale).

        - params_j : List[float]
            Component parameters for component j (skewness, loc, scale).

        - score_min : float | int
            Minimum score to consider when checking for monotonicity.

        - score_max : float | int
            Maximum score to consider when checking for monotonicity.

        Returns
        -------
        bool
            True if the density ratio of component i to component j is non-monotonic, False otherwise.
        """
        score_range = np.linspace(score_min, score_max, 1000)
        log_density_i = skewnorm.logpdf(score_range, *params_i)
        log_density_j = skewnorm.logpdf(score_range, *params_j)
        log_density_ratio = log_density_i - log_density_j
        return not (np.diff(log_density_ratio) <= 0).all()

    def _groupValues(self, vals):
        groups = {}
        for i, val in enumerate(vals):
            if val not in groups:
                groups[val] = []
            groups[val].append(i)
        return groups

    def _groups_from_onehot(self, sampleIndicators: np.ndarray) -> Dict[int, List[int]]:
        """
        Convert one-hot sample indicators to a dictionary of column indices and row indices.
        """
        # Initialize a dictionary to store the column index and list of row indices
        indices_dict = {}

        # Iterate through the sampleIndicators
        for row_idx, row in enumerate(sampleIndicators):
            for col_idx, value in enumerate(row):
                if value == 1:
                    if col_idx not in indices_dict:
                        indices_dict[col_idx] = []
                    indices_dict[col_idx].append(row_idx)
                    break
        return indices_dict

    def _initialize_sample_weights(
        self, scores: np.ndarray, sampleIndicators: np.ndarray
    ) -> None:
        """
        Initialize `sample_weights`.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]


        Returns
        -------
        None
        """
        component_posteriors = self._sample_component_posteriors(
            scores, np.ones(self.num_components) / self.num_components
        )
        self.sample_weights = np.zeros((sampleIndicators.shape[1], self.num_components))
        sample_to_indices = self._groups_from_onehot(sampleIndicators)
        for sampleIdx, indices in sample_to_indices.items():
            self.sample_weights[sampleIdx] = np.mean(
                component_posteriors[indices], axis=0
            )

    def get_component_density(
        self, scores: np.ndarray, component_num: int, **kwargs
    ) -> np.ndarray:
        """
        Get the density of the scores.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - component_num : int
            Component number.

        Optional Parameters
        -------------------
        - log : bool (default False)
            If True, return the log-density of the  scores.
        Returns
        -------
        numpy.array
            Density of the scores.
        """
        if kwargs.get("log", False):
            return skewnorm.logpdf(
                scores,
                self.skewness[component_num],
                self.locs[component_num],
                self.scales[component_num],
            )
        return skewnorm.pdf(
            scores,
            self.skewness[component_num],
            self.locs[component_num],
            self.scales[component_num],
        )

    def get_log_likelihood(
        self, scores: np.ndarray, sampleIndicators: np.ndarray, **kwargs
    ) -> float:
        """
        Get the log_likelihood of the model.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]

        Optional Parameters
        -------------------
        None

        Returns
        -------
        float
            Likelihood of the model.
        """
        LL = 0.0
        for sampleNum, sampleIndices in self._groups_from_onehot(
            sampleIndicators
        ).items():
            if len(sampleIndices) == 0:
                continue
            log_sample_weights = self.sample_weights[sampleNum][..., None]
            log_sample_weights[np.isinf(log_sample_weights)] = 0
            log_densities = np.array(
                [
                    self.get_component_density(scores[sampleIndices], compNum, log=True)
                    for compNum in range(self.num_components)
                ]
            )
            log_likelihood = np.array(
                logsumexp(log_densities + log_sample_weights, axis=0, return_sign=False)
            ).sum()
            if np.isnan(log_likelihood) or np.isinf(log_likelihood):
                raise ValueError("Log likelihood is NaN.")
            LL += log_likelihood
        return LL

    def get_sample_density(self, scores, sampleNum):
        """
        Get the joint density of the scores for the given sample.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - sampleNum : int
            Sample number.

        Returns
        -------
        numpy.array
            Density of the sample.
        """
        density = np.array(
            [
                self.sample_weights[sampleNum, compNum]
                * self.get_component_density(scores, compNum)
                for compNum in range(self.num_components)
                if self.sample_weights[sampleNum, compNum]
            ]
        ).sum(axis=0)
        return density

    def get_sample_cdf(self, scores, sampleNum):
        """
        Get the cumulative distribution function of the sample.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - sampleNum : int
            Sample number.

        Returns
        -------
        numpy.array
            Cumulative distribution function of the sample.
        """
        if sampleNum >= self.sample_weights.shape[0] or sampleNum < 0:
            raise ValueError(
                f"Sample number {sampleNum} must be greater than 0 and less than the number of samples {self.sample_weights.shape[0]}."
            )
        return np.stack(
            [
                self.sample_weights[sampleNum, compNum]
                * self.get_component_cdf(scores, compNum)
                for compNum in range(self.num_components)
            ],
            axis=0,
        ).sum(axis=0)

    def get_component_cdf(self, scores, componentNum):
        """
        Get the cumulative distribution function of the component.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - componentNum : int
            Component number.

        Returns
        -------
        numpy.array
            Cumulative distribution function of the component.
        """
        if componentNum >= self.num_components or componentNum < 0:
            raise ValueError(
                f"Component number {componentNum} must be greater than 0 and less than the number of components {self.num_components}."
            )
        return skewnorm.cdf(
            scores,
            self.skewness[componentNum],
            self.locs[componentNum],
            self.scales[componentNum],
        )

    def get_cdf_distance(self, scores, sampleNum):
        """
        Get the distance between the model's CDF and the empirical CDF of a sample

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - sampleNum : int
            Sample number.

        Returns
        -------
        float
            CDF Distance
        """
        u_scores = sorted(np.unique(scores))
        sample_cdf = self.get_sample_cdf(u_scores, sampleNum)
        empirical_cdf = MulticomponentCalibrationModel.empirical_cdf(u_scores)
        cdf_distance = MulticomponentCalibrationModel.yang_dist(
            sample_cdf, empirical_cdf
        )
        return cdf_distance

    @staticmethod
    def yang_dist(x, y, p=2):
        """
        Normalized metric on functions from 'Yang R, Jiang Y, Mathews S, Housworth EA, Hahn MW, Radivojac P. A new class of metrics for learning on real-valued and structured data. Data Min. Knowl. Disc. (2019) 33(4): 995-1016.'
        d^{2}_N(x,y)

        Parameters
        ----------
        - x : numpy.array
            function x values
        - y : numpy.array
            function y values

        Optional Parameters
        -------------------
        - p : int (default 2)
            p-norm for the metric

        Returns
        -------
        float
            Normalized distance between the functions x and y
        """
        x = np.array(x)
        y = np.array(y)
        gt = x >= y
        dP = ((x[gt] - y[gt]).sum() ** p + (y[~gt] - x[~gt]).sum() ** p) ** (1 / p)
        dPn = dP / sum([max(abs(xi), abs(yi), abs(xi - yi)) for xi, yi in zip(x, y)])
        return dPn

    @staticmethod
    def empirical_cdf(scores):
        """
        Get the empirical cumulative distribution function of the scores.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.

        Returns
        -------
        numpy.array
            Empirical cumulative distribution function of the scores.
        """
        n_scores = len(scores)
        return np.arange(1, n_scores + 1) / n_scores

    def get_component_posteriors(
        self, scores: np.ndarray, sampleIndicators: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Get the component posteriors for each sample.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - sampleIndicators : numpy.array
            One-hot sample indicators, e.g., columns [0,1,2,3] -> [benign, pathogenic, gnomAD, synonymous]

        Optional Parameters
        -------------------
        - logsumexp : bool (default False)
            If True, use logsumexp to compute the posterior probabilities.

        Returns
        -------
        numpy.array
            Posterior probabilities of each sample being from each component.
        """
        NObservations, NSamples = sampleIndicators.shape
        _, NComponents = self.sample_weights.shape
        if NObservations != scores.shape[0]:
            raise ValueError(
                f"The number of observations {NObservations} must match the number of scores {scores.shape[0]}."
            )
        if NSamples != self.sample_weights.shape[0]:
            raise ValueError(
                f"The number of samples {NSamples} must match the number of sample weights {self.sample_weights.shape[0]}."
            )
        comp_posteriors = np.zeros((NObservations, NComponents))
        for sampleNum, sampleIndices in self._groups_from_onehot(
            sampleIndicators
        ).items():
            comp_posteriors[sampleIndices] = self._sample_component_posteriors(
                scores[sampleIndices], self.sample_weights[sampleNum], **kwargs
            )
        return comp_posteriors

    def _sample_component_posteriors(
        self, sample_scores: np.ndarray, sample_weights: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Get the posterior probabilities of each scores.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - weights : numpy.array
            Weight of each component for the given mixture

        Optional Parameters
        -------------------
        - logsumexp : bool (default True)
            If True, use logsumexp to compute the posterior probabilities.
        Returns
        -------
        numpy.array
            Posterior probabilities of each scores. (shape: [n_samples, n_components])
        """
        if len(sample_weights) != self.num_components:
            raise ValueError(
                f"The number of sample weights {len(sample_weights)} must match the number of components {self.num_components}."
            )
        sample_weights = np.array(sample_weights)[:, None]
        comp_posteriors = np.zeros((sample_scores.shape[0], self.num_components))
        if kwargs.get("logsumexp", True):
            log_pdfs = np.stack(
                [
                    self.get_component_density(sample_scores.ravel(), i, log=True)
                    for i in range(self.num_components)
                ],
                axis=0,
            )
            numerators = np.zeros_like(log_pdfs)

            # numerators = log_pdfs + np.log(sample_weights)
            for compNum in range(self.num_components):
                if sample_weights[compNum] == 0:
                    continue
                numerators[compNum] = log_pdfs[compNum] + np.log(
                    sample_weights[compNum]
                )
            d = logsumexp(numerators, axis=0)
            comp_posteriors = np.exp(numerators - d[None])  # type: ignore
            comp_posteriors[np.isnan(comp_posteriors)] = 0
            comp_posteriors = comp_posteriors.T
        else:
            for componentNum in range(self.num_components):
                comp_posteriors[:, componentNum] = (
                    self.get_component_density(sample_scores, componentNum)
                    * sample_weights[componentNum]
                )
            comp_posteriors /= np.sum(comp_posteriors, axis=1)[:, np.newaxis]
        assert np.allclose(
            np.sum(comp_posteriors, axis=1), 1
        ), f"Posterior probabilities must sum to 1.\n{comp_posteriors.sum(axis=1)}"
        return comp_posteriors

    def predict(self, scores, sampleIndicators, **kwargs):
        """
        Predict the probability of each component for each score

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., [benign, pathogenic, gnomAD, synonymous

        Returns
        -------
        numpy.array
            Predicted probability of each component for each score
        """
        self.validate_inputs(scores, sampleIndicators)
        return self.get_component_posteriors(scores, sampleIndicators, **kwargs)

    def fit_predict(self, scores, sampleIndicators, **kwargs):
        """
        Fit the model to the given assay scores and sample indicators, and predict the probability of each sample being generated by each of the components.

        Parameters
        ----------
        scores : numpy.array
            Assay scores.
        sampleIndicators : numpy.array
            One-hot sample indicators, e.g., [benign, pathogenic, gnomAD, synonymous]

        Returns
        -------
        numpy.array
            Predicted probabilities of each sample being generated by each of the components.
        """
        # fit the model to the given assay scores and sample indicators
        self.fit(scores, sampleIndicators)
        # predict the probability of each sample being generated by each of the components
        return self.predict(scores, sampleIndicators, **kwargs)

    def get_params(self):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        None

        Returns
        -------
        dict
            Parameters for this estimator.
        """
        params = dict(
            skewness=self.skewness,
            locs=self.locs,
            scales=self.scales,
            sample_weights=self.sample_weights,
        )
        return {k: v.copy() for k, v in params.items()}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Returns
        -------
        self
        """
        for k, v in params.items():
            setattr(self, k, v)

    def get_metadata_routing(self) -> None:
        """
        For compatibility with scikit-learn models

        Returns
        -------
        None
        """
        pass

    def binary_search(
        self,
        scores,
        candiate_value,
        previous_canonical_param_pair,
        parameter_idx,
        component_num,
        **kwargs,
    ):
        """
        Run binary search to find the parameter value that satisfies monotonicity.

        Parameters
        ----------
        - scores : numpy.array
            Assay scores.
        - candidate_value : float
            Proposed parameter value in alternate parameterization (i.e., location, Delta, Gamma).
        - previous_canonical_param_pair : Tuple[Tuple[float, float, float], Tuple[float, float, float]]
            Pair of previous component parameters in canonical parameterization (i.e., skewness, loc, scale).
        - updating_first_in_tuple : bool
            If True, update the first component in the tuple, otherwise update the second component.
        - parameter_idx : int
            Index of the parameter to update in the alternate parameterization (i.e., 0 -> location, 1 -> Delta, 2 -> Gamma).

        Returns
        -------
        float
            Updated parameter value in alternate parameterization.

        """
        unique_scores = np.unique(scores)
        unique_scores.sort()
        # Get the alternate parameterization for the previous component parameters (i.e., location, Delta, Gamma)
        if self._current_component != self.num_components - 1:
            assert not MulticomponentCalibrationModel.parameters_violate_monotonicity(
                previous_canonical_param_pair[self._current_component],
                previous_canonical_param_pair[self._current_component + 1],
                self.score_min,
                self.score_max,
            )
        if self._current_component != 0:
            assert not MulticomponentCalibrationModel.parameters_violate_monotonicity(
                previous_canonical_param_pair[self._current_component - 1],
                previous_canonical_param_pair[self._current_component],
                self.score_min,
                self.score_max,
            )
        previous_alternate_param_pair = [
            list(self.canonical_to_alternate(*param))
            for param in previous_canonical_param_pair
        ]
        # Get the lower bound for the binary search, i.e., the previous parameter value
        lower_bound = previous_alternate_param_pair[self._current_component][
            parameter_idx
        ]
        # Get the upper bound for the binary search, i.e., the candidate parameter value
        upper_bound = candiate_value
        updated_params = [
            list(deepcopy(previous_alternate_param))
            for previous_alternate_param in previous_alternate_param_pair
        ]
        while abs(upper_bound - lower_bound) > self._tol:
            # Get the midpoint of the lower and upper bounds
            midpoint = (upper_bound + lower_bound) / 2
            # Update the parameter value in the alternate parameterization
            updated_params[self._current_component][parameter_idx] = midpoint
            # if updating the first component, check monotonicity with the second component
            if self._current_component == 0:
                if MulticomponentCalibrationModel.parameters_violate_monotonicity(
                    self.alternate_to_canonical(*updated_params[0]),
                    self.alternate_to_canonical(*updated_params[1]),
                    self.score_min,
                    self.score_max,
                ):
                    # midpoint violates monotonicity, reduce the upper bound
                    upper_bound = midpoint
                else:
                    # Otherwise, increase the lower bound
                    lower_bound = midpoint
            # if checking monotonicity with the last component, check monotonicity with the second-to-last component
            elif self._current_component == self.num_components - 1:
                if MulticomponentCalibrationModel.parameters_violate_monotonicity(
                    self.alternate_to_canonical(
                        *updated_params[self._current_component - 1]
                    ),
                    self.alternate_to_canonical(
                        *updated_params[self._current_component]
                    ),
                    self.score_min,
                    self.score_max,
                ):
                    # midpoint violates monotonicity, reduce the upper bound
                    upper_bound = midpoint
                else:
                    # Otherwise, increase the lower bound
                    lower_bound = midpoint
            # otherwise, check monotonicity with the previous and next components
            else:
                if MulticomponentCalibrationModel.parameters_violate_monotonicity(
                    self.alternate_to_canonical(
                        *updated_params[self._current_component - 1]
                    ),
                    self.alternate_to_canonical(
                        *updated_params[self._current_component]
                    ),
                    self.score_min,
                    self.score_max,
                ) or MulticomponentCalibrationModel.parameters_violate_monotonicity(
                    self.alternate_to_canonical(
                        *updated_params[self._current_component]
                    ),
                    self.alternate_to_canonical(
                        *updated_params[self._current_component + 1]
                    ),
                    self.score_min,
                    self.score_max,
                ):
                    # midpoint violates monotonicity, reduce the upper bound
                    upper_bound = midpoint
                else:
                    # Otherwise, increase the lower bound
                    lower_bound = midpoint
        updated_params[self._current_component][parameter_idx] = lower_bound
        for compI in range(self.num_components):
            canonI = self.alternate_to_canonical(*updated_params[compI])
            if compI != self.num_components - 1:
                canonJ = self.alternate_to_canonical(*updated_params[compI + 1])
                if MulticomponentCalibrationModel.parameters_violate_monotonicity(
                    canonI,
                    canonJ,
                    self.score_min,
                    self.score_max,
                ):
                    raise ValueError(
                        f"Component {compI} and {compI+1} violate monotonicity after binary search for {self._current_component}."
                    )
            if compI != 0:
                canonJ = self.alternate_to_canonical(*updated_params[compI - 1])
                if MulticomponentCalibrationModel.parameters_violate_monotonicity(
                    canonJ,
                    canonI,
                    self.score_min,
                    self.score_max,
                ):
                    raise ValueError(
                        f"Component {compI-1} and {compI} violate monotonicity after binary search for {self._current_component}."
                    )
                
            # print(f"Component {compI} and {compJ} are monotonic.")
        # print(f"All monotoniciy constraints satisfied binary search of parameter {parameter_idx} for component {component_num}.")
        return lower_bound
