RetailHero Uplift Modeling
==========================

Uplift-modelling task. Rank the customers in descending order of communication efficiency.

The competition page: https://retailhero.ai/c/uplift_modeling/

**The data**:
* data/clients.csv — clients information
* data/products.csv — products information
* data/purchases.csv — purchase history before sms advertisement
* data/uplift_train.csv — train set of clients with conversion information
* data/uplift_test.csv — test set to model the uplift

**Solutions**:
* Baseline solution: `uplift_solution.py`
* Final solution **(top-100)**: `create_dataset.py`, `pipeline.ipynb`

The libraries needed to run the code are listed in `requirements.txt`.

**Reference materials**:
* https://github.com/maks-sh/scikit-uplift
* https://github.com/olegitor/X5.Uplift.public
* https://github.com/nersirion/nersirion-RetailHero.ai-uplift
* https://github.com/rekcahd/retailhero_uplift

**Tutorials on uplift-modeling (in Russian)**:
* [Туториал по Uplift моделированию. Часть 1](https://datafest.us18.list-manage.com/track/click?u=acc56a45f4f4d03aa67f9cd69&id=105e33fc89&e=e9d30d7bbb)
* [Туториал по Uplift моделированию. Часть 2](https://datafest.us18.list-manage.com/track/click?u=acc56a45f4f4d03aa67f9cd69&id=dd31184da8&e=e9d30d7bbb)
