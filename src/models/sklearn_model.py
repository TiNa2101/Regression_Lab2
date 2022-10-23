import category_encoders as ce
import src.config as cfg
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

real_pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
    ]
)

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]
)

preprocess_pipe = ColumnTransformer(transformers=[
    ('real_cols', real_pipe, cfg.REAL_COLS),
    ('cat_cols', cat_pipe, cfg.CAT_COLS),
    ('woe_cat_cols', ce.WOEEncoder(), cfg.CAT_COLS),
    ('ohe_cols', 'passthrough', cfg.OHE_COLS)
    ]
)

base_model = LinearSVC(max_iter=10000)

model = MultiOutputClassifier(Pipeline([
    ('preprocess', preprocess_pipe),
    ('model', base_model)
    ]
    )
)
