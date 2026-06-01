from typing import Optional

from autofit.database.sqlalchemy_ import sa
from autofit.non_linear.initializer import AbstractInitializer
from searches.pyswarms.abstract import AbstractPySwarms


class PySwarmsGlobal(AbstractPySwarms):
    __identifier_fields__ = (
        "n_particles",
        "cognitive",
        "social",
        "inertia",
    )

    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        initializer: Optional[AbstractInitializer] = None,
        iterations_per_full_update: int = None,
        iterations_per_quick_update: int = None,
        number_of_cores: int = 1,
        silence: bool = False,
        session: Optional[sa.orm.Session] = None,
        **kwargs
    ):
        """
        A PySwarms Particle Swarm MLE global-best non-linear search.

        For a full description of PySwarms, checkout its Github and readthedocs webpages:

        https://github.com/ljvmiranda921/pyswarms

        https://pyswarms.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        number_of_cores
            The number of cores sampling is performed using a Python multiprocessing Pool instance.
        silence
            If True, the default print output of the non-linear search is silenced.
        """

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            initializer=initializer,
            iterations_per_quick_update=iterations_per_quick_update,
            iterations_per_full_update=iterations_per_full_update,
            number_of_cores=number_of_cores,
            silence=silence,
            session=session,
            **kwargs
        )

        self.logger.debug("Creating PySwarms Search")

    def search_internal_from(self, model, fitness, bounds, init_pos):
        """Get the PySwarms GlobalBestPSO sampler which performs the non-linear search."""

        import pyswarms

        options = {
            "c1": self.cognitive,
            "c2": self.social,
            "w": self.inertia,
        }

        return pyswarms.global_best.GlobalBestPSO(
            n_particles=self.n_particles,
            dimensions=model.prior_count,
            bounds=bounds,
            init_pos=init_pos,
            options=options,
        )
