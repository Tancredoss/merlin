.. _user_guide_compatibility:

Compatibility
=============

Use the supported version combinations below when installing or upgrading MerLin.

.. list-table::
   :header-rows: 1
   :widths: 16 20 20 20 16 18

   * - MerLin version
     - Perceval version
     - PyTorch version
     - scikit-learn version
     - Python version
     - Support status
   * - ``0.4``
     - ``>=1.2.1``
     - ``>=2.0.0, <2.13``
     - ``>=1.7.2, <1.10``
     - ``>=3.10``
     - Current release line.
   * - ``<=0.3``
     - ``>=0.13.1, <=1.1``
     - ``>=2.0.0, <=2.10.0``
     - ``>=1.7.2, <1.10``
     - ``>=3.10``
     - Legacy Perceval API support.

.. note::

   Some MerLin ``<=0.3`` package metadata may not fully reflect the intended
   upper bounds for PyTorch, Perceval and scikit-learn. Use this table as the reference when
   selecting dependency versions for older MerLin releases.
