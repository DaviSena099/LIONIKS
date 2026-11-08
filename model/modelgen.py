
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Dataset
dataset = pd.read_csv("data/dataset/dataset.csv")

X = dataset.drop(columns=[
    'family', 'name', 'symbol', 'electron_configuration'
])

y = dataset["family"]

# Identificação das colunas
categorical_features = [
    'block',
    'state_25C',
    'type'
]

numeric_features = X.select_dtypes(
    include=['number']
).columns.tolist()

# Pré-processamento
preprocessor = ColumnTransformer(
    transformers=[
        (
            'categorical',
            OneHotEncoder(handle_unknown='ignore'),
            categorical_features
        ),
        (
            'numeric',
            'passthrough',
            numeric_features
        )
    ]
)

# Modelo
model = RandomForestClassifier(
    n_estimators=300,
    criterion="entropy",
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Separação dos dados
train_X, test_X, train_y, test_y = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Validação cruzada
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_validate(
    pipeline,
    train_X,
    train_y,
    cv=cv,
    scoring=[
        'accuracy',
        'f1_macro',
        'precision_macro',
        'recall_macro'
    ],
    n_jobs=-1
)

print("\nCROSS VALIDATION")

for metric in scores:
    if metric.startswith('test_'):
        print(
            f"{metric}: "
            f"{scores[metric].mean():.4f} "
            f"+/- {scores[metric].std():.4f}"
        )

# Treinamento final
pipeline.fit(train_X, train_y)

# Avaliação no conjunto de teste
predictions = pipeline.predict(test_X)

print("\nCLASSIFICATION REPORT")
print(classification_report(test_y, predictions))

print("\nCONFUSION MATRIX")
print(confusion_matrix(test_y, predictions))

# Salvamento
joblib.dump(pipeline, 'model/lioniks_rfc.pkl')
