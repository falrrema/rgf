Faster exact RGF training and reliable model saving
==================================================

This fork preserves the original RGF, RGF_Opt and RGF_Sib algorithms. RGF_Sib
keeps its sibling constraint and corrective weight optimization; there is no
switch to FastRGF. Three training changes make the native trainer faster and
smaller, and every one of them produces the same model bytes and the same
float64 predictions as the original code.

Parallel split search (``split_threads``)
-----------------------------------------

Each worker searches a contiguous range of features using the original code.
Each feature keeps the original row order, floating-point calculations and
threshold order. Workers have separate candidates, cursors and temporary sorted
feature buffers. The training thread combines their results in feature order,
using the original strict greater-than comparison, so equal scores keep the same
first feature and threshold as before.

The worker pool persists between searches. Nodes with fewer than 32,768
row-feature pairs stay serial to avoid dispatch overhead. The same pool copies
sorted feature indexes and stably partitions them into child nodes; each feature
owns its output storage and the shared membership mask is read-only. Tree
growth, regularization, residual updates and corrective optimization remain
sequential. No reduced precision, fast-math flags, approximation or training-row
sampling is introduced. Sparse and conservative memory modes keep their original
behavior.

``split_threads`` is a positive integer, defaults to one, and is accepted by both
the native command line and the Python wrapper. At one worker the wrapper omits
the parameter, so it stays compatible with an unmodified executable. Classifier
``n_jobs`` still controls separate classification processes, each of which can
create its own split workers; set both deliberately to avoid oversubscription.

Split-sum cache (``split_cache``)
---------------------------------

``RGF_Sib`` and ``RGF_Opt`` rescore every leaf of the searched tree each time a
leaf is added ("refresh all"). Most of those leaves have not changed: their rows
keep the same targets until the next weight correction or a split in another
tree. Each leaf now records the left/right target sums of every split it
evaluated, and an unchanged leaf replays those sums with the current penalty
terms instead of rescanning its rows. Records are discarded when the leaf splits,
when weights are corrected, when another tree splits, and whenever feature
sampling (``f_ratio``) is on. On by default; ``split_cache=0`` restores
rescanning.

One-byte level codes (``split_histogram``)
------------------------------------------

When the data is managed as dense, has no NaN and every feature has at most 256
distinct values, each value is kept as one byte: its rank among that feature's
own distinct values. Split search adds up each value's targets over a node's
rows instead of walking per-feature sorted row lists. The double-precision copy
of the matrix, the initial serial sort and all sorted lists are no longer built.

This is an exact histogram, not binning: the levels are the data's own values,
thresholds are the original midpoints, each level adds its rows in the order the
sorted list held them (value, then row), and children receive their rows in that
same order. Other data, such as continuous features or data managed as sparse,
uses the sorted lists automatically. On by default; ``split_histogram=0`` forces
the sorted lists. The training input is also handed from the reader to the
trainer one column at a time instead of being copied whole, and freed memory is
returned to the operating system after setup.

Measured training speed
-----------------------

Synthetic fixture: integer features 0--4, 256 features, ``RGF_Sib`` to 512
leaves, learning rate 0.01, optimization and checkpoint intervals of 128 leaves.
Median of three alternating-order repetitions on a shared 16-logical-CPU Apple
arm64 machine. Times include native process startup, reading the input, sorting,
training and writing all checkpoints.

Parallel split search alone, 16,384 rows:

======================= =============== ===================
Configuration           Median seconds  Range across 3 runs
======================= =============== ===================
Unchanged original      15.378          14.373 to 15.405
Updated, one worker     14.684          13.877 to 14.843
Updated, four workers   11.050          10.803 to 11.142
Updated, eight workers  10.291           9.961 to 10.569
======================= =============== ===================

Parallel sorted-row preparation, eight workers in both columns:

=========== ================== ==================
Rows        Split search only  Plus preparation
=========== ================== ==================
16,384      12.469 s           8.499 s
65,536      51.482 s           23.951 s
=========== ================== ==================

Split-sum cache (stage 1) and level codes (stage 2), eight workers:

=========== ============== ============== ============== ==========================
Rows        Before         Stage 1        Stage 2        Peak resident memory
=========== ============== ============== ============== ==========================
16,384      4.11 s         2.99 s         1.58 s         244 MB to 88 MB
65,536      16.86 s        12.16 s        6.27 s         933 MB to 329 MB
=========== ============== ============== ============== ==========================

On larger inputs with many rows and features, level codes reduced peak memory by
roughly two thirds and setup time by about ten times, because the
double-precision matrix and the per-feature sorted lists are no longer built.
The split-sum cache helps most when the forest grows few, large trees; with many
small trees each refresh replays only a few leaves. Only macOS was built and
tested.

The native ``Time`` diagnostic reports CPU time, so it should not be confused
with elapsed time when several workers run at once.

Verification
------------

Each change was compared with an independently built copy of the unchanged
commit ``272afb85b4c91571f576e5fc83ecfacce3672eb4``. The comparisons cover all
native model bytes except configuration and signature text (tree topology,
thresholds, weights, split gains, parent pointers, feature count and intercept),
and predictions in their full 64-bit representation through the original native
predictor. Every checkpoint and every prediction matched exactly.

Coverage includes RGF, RGF_Opt and RGF_Sib; one, two, four and eight workers;
repeated fits; weighted training; warm starts; ``num_tree_search`` above one;
the conservative memory mode; feature sampling; frequent correction; continuous,
sparse and constant features; signed zeros; a 256-level feature and a 257-level
feature (automatic fallback); exact ties across worker ranges; query values
immediately below, on and above learned thresholds; and squared, absolute,
logistic and exponential losses.

Usage
-----

Build the native executable from the repository root::

    cmake -S RGF -B /tmp/rgf-build
    cmake --build /tmp/rgf-build --parallel 4

The executable is written to ``RGF/bin/rgf``. To use the source wrapper with it::

    export PYTHONPATH="$PWD/python-package${PYTHONPATH:+:$PYTHONPATH}"
    export PATH="$PWD/RGF/bin:$PATH"

If an installed wrapper or another RGF executable takes precedence in your
setup, check ``rgf.__file__`` and ``rgf.utils.Config().RGF_PATH`` before
training. For example::

    import pickle
    from rgf.sklearn import RGFRegressor

    model = RGFRegressor(
        algorithm="RGF_Sib",
        max_leaf=512,
        test_interval=128,
        opt_interval=128,
        learning_rate=0.01,
        split_threads=8,
    ).fit(X_train, y_train)

    with open("model.pkl", "wb") as stream:
        pickle.dump(model, stream)

    with open("model.pkl", "rb") as stream:
        restored = pickle.load(stream)
    predictions = restored.predict(X_test)

Saving and checkpoint selection
-------------------------------

A fitted pickle embeds the selected native model. Loading it resolves the
receiving runtime's configuration, creates a unique temporary filename and
restores the embedded bytes there. It no longer writes to the original machine's
filename or replaces the current global configuration with the pickled one.
Older pickles remain supported. Separate loads have separate cleanup ownership,
including when the temporary-directory setting changes later.

Unfitted executers passed to local classification workers keep their explicit
training configuration, which custom executable locations require. Cleanup
records each executer's own directory instead of rediscovering binaries at
worker exit.

``save_model(filename)`` exports the native model for warm starts and now returns
the estimator as documented. It is distinct from pickling the whole estimator.

Checkpoint selection accepts only numeric checkpoint files and chooses the
largest numeric suffix, so checkpoint 100 takes precedence over 99; log files,
backup files and directories are ignored. This selects the latest checkpoint,
not the one with the best validation score.

A standard estimator pickle still requires the Python wrapper and a compatible
native executable on the receiving machine.

To run the wrapper tests::

    PYTHONPATH="$PWD/python-package" RGF_BINARY="$PWD/RGF/bin/rgf" \
      python -m unittest discover -s python-package/tests \
      -p test_persistence_parallel.py -v

Set ``RGF_ORIGINAL_PACKAGE`` to an unmodified ``python-package`` directory to
also run the old-pickle compatibility test.
