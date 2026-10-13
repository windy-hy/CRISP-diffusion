import numpy as np


class MeanShiftBaseline:
    """
    Simple non-parametric mean-shift baseline.

    No neural-network training.

    Training/statistics stage:
        delta = mean(treated_train) - mean(control_train)

    Prediction:
        pred = target_control + delta
    """

    def __init__(self, clip_nonnegative=True):
        self.shift_dict = {}
        self.clip_nonnegative = clip_nonnegative

    def fit(
        self,
        key,
        treated_groups,
        control_groups,
    ):
        """
        Parameters
        ----------
        key:
            Usually (drug, dose)

        treated_groups:
            list of arrays
            Each array corresponds to one training context/cell type:
                [n_cells, n_genes]

        control_groups:
            matched list of control arrays.

        Important:
            We first calculate one shift per training group,
            then average the shifts.

            This avoids large cell populations dominating the baseline.
        """

        if len(treated_groups) != len(control_groups):
            raise ValueError(
                "treated_groups and control_groups must have the same length"
            )

        if len(treated_groups) == 0:
            raise ValueError(f"No training groups available for key={key}")

        shifts = []

        for treated, control in zip(
            treated_groups,
            control_groups
        ):
            treated = np.asarray(
                treated,
                dtype=np.float32
            )
            control = np.asarray(
                control,
                dtype=np.float32
            )

            if treated.ndim != 2 or control.ndim != 2:
                raise ValueError(
                    "treated/control must be [n_cells, n_genes]"
                )

            shift = (
                treated.mean(axis=0)
                - control.mean(axis=0)
            )

            shifts.append(shift)

        # Equal weight for each training context
        mean_shift = np.mean(
            np.stack(shifts, axis=0),
            axis=0
        )

        self.shift_dict[key] = mean_shift

    def predict(
        self,
        key,
        control,
    ):
        """
        Parameters
        ----------
        control:
            target control population
            [n_cells, n_genes]

        Returns
        -------
        pred:
            predicted treated population
            [n_cells, n_genes]
        """

        if key not in self.shift_dict:
            raise KeyError(
                f"No mean shift available for {key}"
            )

        control = np.asarray(
            control,
            dtype=np.float32
        )

        shift = self.shift_dict[key]

        pred = (
            control
            + shift[None, :]
        )

        if self.clip_nonnegative:
            pred = np.maximum(
                pred,
                0.0
            )

        return pred