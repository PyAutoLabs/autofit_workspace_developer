import os
from typing import Dict, Optional

from autofit.database.sqlalchemy_ import sa

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.nest import abstract_nest
from autofit.non_linear.fitness import Fitness
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.nest import SamplesNest
from autofit.non_linear.test_mode import is_test_mode


class UltraNest(abstract_nest.AbstractNest):
    __identifier_fields__ = (
        "draw_multiple",
        "ndraw_min",
        "ndraw_max",
        "min_num_live_points",
        "cluster_num_live_points",
        "insertion_test_zscore_threshold",
        "stepsampler_cls",
        "nsteps",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        draw_multiple: bool = True,
        ndraw_min: int = 128,
        ndraw_max: int = 65536,
        num_bootstraps: int = 30,
        num_test_samples: int = 2,
        resume: bool = True,
        run_num: Optional[int] = None,
        storage_backend: str = "hdf5",
        vectorized: bool = False,
        warmstart_max_tau: float = -1.0,
        min_num_live_points: int = 400,
        cluster_num_live_points: int = 40,
        insertion_test_window: int = 10,
        insertion_test_zscore_threshold: int = 2,
        dlogz: float = 0.5,
        dkl: float = 0.5,
        frac_remain: float = 0.01,
        lepsilon: float = 0.001,
        min_ess: int = 400,
        max_iters: Optional[int] = None,
        max_ncalls: Optional[int] = None,
        max_num_improvement_loops: float = -1.0,
        log_interval: Optional[int] = None,
        show_status: bool = True,
        update_interval_ncall: Optional[int] = None,
        update_interval_volume_fraction: float = 0.8,
        viz_callback: str = "auto",
        stepsampler_cls: Optional[str] = None,
        nsteps: int = 25,
        adaptive_nsteps: bool = False,
        log: bool = False,
        max_nsteps: int = 1000,
        region_filter: bool = False,
        scale: float = 1.0,
        iterations_per_quick_update: int = None,
        iterations_per_full_update: int = None,
        number_of_cores: int = 1,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs
    ):
        """
        An UltraNest non-linear search.

        UltraNest is an optional requirement and must be installed manually via the command `pip install ultranest`.
        It is optional as it has certain dependencies which are generally straight forward to install (e.g. Cython).

        For a full description of UltraNest and its Python wrapper PyUltraNest, checkout its Github and documentation
        webpages:

        https://github.com/JohannesBuchner/UltraNest
        https://johannesbuchner.github.io/UltraNest/readme.html

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        iterations_per_full_update
            The number of iterations performed between update (e.g. output latest model to hard-disk, visualization).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.
        silence
            If True, the default print output of the non-linear search is silenced.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs
        )

        self.draw_multiple = draw_multiple
        self.ndraw_min = ndraw_min
        self.ndraw_max = ndraw_max
        self.num_bootstraps = num_bootstraps
        self.num_test_samples = num_test_samples
        self.resume = resume
        self.run_num = run_num
        self.storage_backend = storage_backend
        self.vectorized = vectorized
        self.warmstart_max_tau = warmstart_max_tau

        self.min_num_live_points = min_num_live_points
        self.cluster_num_live_points = cluster_num_live_points
        self.insertion_test_window = insertion_test_window
        self.insertion_test_zscore_threshold = insertion_test_zscore_threshold
        self.dlogz = dlogz
        self.dkl = dkl
        self.frac_remain = frac_remain
        self.lepsilon = lepsilon
        self.min_ess = min_ess
        self.max_iters = max_iters
        self.max_ncalls = max_ncalls
        self.max_num_improvement_loops = max_num_improvement_loops
        self.log_interval = log_interval
        self.show_status = show_status
        self.update_interval_ncall = update_interval_ncall
        self.update_interval_volume_fraction = update_interval_volume_fraction
        self.viz_callback = viz_callback

        self.stepsampler_cls = stepsampler_cls
        self.nsteps = nsteps if stepsampler_cls is not None else None
        self.adaptive_nsteps = adaptive_nsteps
        self.log_stepsampler = log
        self.max_nsteps = max_nsteps
        self.region_filter = region_filter
        self.scale = scale

        if is_test_mode():
            self.apply_test_mode()

        self.logger.debug("Creating UltraNest Search")

    def apply_test_mode(self):
        self.max_iters = 1
        self.max_ncalls = 1

    @property
    def search_kwargs(self):
        """Build the kwargs dict passed to ``ReactiveNestedSampler``."""
        return {
            "draw_multiple": self.draw_multiple,
            "ndraw_min": self.ndraw_min,
            "ndraw_max": self.ndraw_max,
            "num_bootstraps": self.num_bootstraps,
            "num_test_samples": self.num_test_samples,
            "resume": self.resume,
            "run_num": self.run_num,
            "storage_backend": self.storage_backend,
            "vectorized": self.vectorized,
            "warmstart_max_tau": self.warmstart_max_tau,
        }

    @property
    def run_kwargs(self):
        """Build the kwargs dict passed to ``sampler.run()``."""
        return {
            "min_num_live_points": self.min_num_live_points,
            "cluster_num_live_points": self.cluster_num_live_points,
            "insertion_test_window": self.insertion_test_window,
            "insertion_test_zscore_threshold": self.insertion_test_zscore_threshold,
            "frac_remain": self.frac_remain,
            "min_ess": self.min_ess,
            "max_iters": self.max_iters,
            "max_num_improvement_loops": self.max_num_improvement_loops,
            "log_interval": self.log_interval,
            "show_status": self.show_status,
            "update_interval_volume_fraction": self.update_interval_volume_fraction,
            "viz_callback": self.viz_callback,
        }

    def _fit(self, model: AbstractPriorModel, analysis):
        """
        Fit a model using the search and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the `NonLinearSearch` seeks to maximize.

        Parameters
        ----------
        model : ModelMapper
            The model which generates instances for different points in parameter space.
        analysis : Analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.

        Returns
        -------
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted ssamples of the fit.
        """

        try:
            import ultranest
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "\n--------------------\n"
                "You are attempting to perform a model-fit using UltraNest. \n\n"
                "However, the optional library UltraNest (https://johannesbuchner.github.io/UltraNest/index.html) is "
                "not installed.\n\n"
                "Install it via the command `pip install ultranest==3.6.2`.\n\n"
                "----------------------"
            )

        fitness = Fitness(
            model=model,
            analysis=analysis,
            paths=self.paths,
            fom_is_log_likelihood=True,
            resample_figure_of_merit=-1.0e99,
        )

        def prior_transform(cube):
            return model.vector_from_unit_vector(
                unit_vector=cube,
            )

        log_dir = self.paths.search_internal_path

        try:
            checkpoint_exists = Path(log_dir / "chains").exists()
        except TypeError:
            checkpoint_exists = False

        if checkpoint_exists:
            self.logger.info(
                "Resuming UltraNest non-linear search (previous samples found)."
            )
        else:
            self.logger.info(
                "Starting new UltraNest non-linear search (no previous samples found)."
            )

        search_internal = ultranest.ReactiveNestedSampler(
            param_names=model.parameter_names,
            loglike=fitness.call_wrap,
            transform=prior_transform,
            log_dir=log_dir,
            **self.search_kwargs
        )

        search_internal.stepsampler = self.stepsampler

        finished = False

        while not finished:

            try:
                total_iterations = search_internal.ncall
            except AttributeError:
                total_iterations = 0

            if self.max_ncalls is not None:
                iterations = self.max_ncalls
            else:
                iterations = total_iterations + self.iterations_per_full_update

            if iterations > 0:

                run_kwargs = self.run_kwargs
                run_kwargs["update_interval_ncall"] = iterations

                search_internal.run(max_ncalls=iterations, **run_kwargs)

            self.paths.save_search_internal(
                obj=search_internal.results,
            )

            iterations_after_run = search_internal.ncall

            if (
                total_iterations == iterations_after_run
                or iterations_after_run == self.max_ncalls
            ):
                finished = True

            if not finished:

                self.perform_update(
                    model=model,
                    analysis=analysis,
                    during_analysis=True,
                    fitness=fitness,
                    search_internal=search_internal,
                )

        return search_internal, fitness

    def output_search_internal(self, search_internal):
        """
        Output the sampler results to hard-disk in their internal format.

        UltraNest uses a backend to store and load results, therefore the outputting of the search internal to a
        dill file is disabled.

        However, a dictionary of the search results is output to dill above.

        Parameters
        ----------
        sampler
            The nautilus sampler object containing the results of the model-fit.
        """
        pass

    def samples_info_from(self, search_internal=None):

        search_internal = search_internal or self.paths.load_search_internal()

        return {
            "log_evidence": search_internal["logz"],
            "total_samples": search_internal["ncall"],
            "total_accepted_samples": len(search_internal["weighted_samples"]["logl"]),
            "time": self.timer.time if self.timer else None,
            "number_live_points": self.min_num_live_points,
        }

    def samples_via_internal_from(
        self, model: AbstractPriorModel, search_internal=None
    ):
        """
        Returns a `Samples` object from the ultranest internal results.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        search_internal = search_internal.results or self.paths.load_search_internal()

        parameters = search_internal["weighted_samples"]["points"]
        log_likelihood_list = search_internal["weighted_samples"]["logl"]
        log_prior_list = [
            sum(model.log_prior_list_from_vector(vector=vector))
            for vector in parameters
        ]
        weight_list = search_internal["weighted_samples"]["weights"]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameters,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return SamplesNest(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info_from(search_internal=search_internal),
        )

    @property
    def stepsampler(self):

        from ultranest import stepsampler

        stepsampler_cls = self.stepsampler_cls

        if stepsampler_cls is None:
            return None

        stepsampler_kwargs = {
            "nsteps": self.nsteps,
            "adaptive_nsteps": self.adaptive_nsteps,
            "log": self.log_stepsampler,
            "max_nsteps": self.max_nsteps,
            "region_filter": self.region_filter,
            "scale": self.scale,
        }

        if stepsampler_cls == "RegionMHSampler":
            return stepsampler.RegionMHSampler(**stepsampler_kwargs)
        elif stepsampler_cls == "AHARMSampler":
            stepsampler_kwargs.pop("scale")
            return stepsampler.AHARMSampler(**stepsampler_kwargs)
        elif stepsampler_cls == "CubeMHSampler":
            return stepsampler.CubeMHSampler(**stepsampler_kwargs)
        elif stepsampler_cls == "CubeSliceSampler":
            return stepsampler.CubeSliceSampler(**stepsampler_kwargs)
        elif stepsampler_cls == "RegionSliceSampler":
            return stepsampler.RegionSliceSampler(**stepsampler_kwargs)
