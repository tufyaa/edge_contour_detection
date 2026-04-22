# Проект: Edge and Contour Detection

## Краткая цель

Проект `ippi-edges` - Python-пакет и CLI-приложение для пакетной обработки изображений из папки:

- выделение краев оператором Sobel;
- выделение краев оператором Laplacian;
- построение бинарной карты краев;
- поиск и отрисовка контуров;
- сохранение результатов и сводной таблицы по каждому изображению.

Тема соответствует варианту:

> Edge and Contour Detection. For each image in a folder, detect edges and contours. Key techniques: Sobel operator, Laplacian operator.

Проект должен быть не набором скриптов, а полноценным пакетом: `uv`, `pyproject.toml`, `src` layout, `tests`, `click` CLI, `pytest`, `ruff`, type checking, Sphinx-документация, `poe` run targets, CI/CD, отчет по производительности.

## Что берется из эталона `hypermodern_python`

Эталонный проект задает стиль, который нужно повторить:

- конфигурация проекта живет в `pyproject.toml`;
- код пакета лежит в `src/<module_name>`;
- тесты лежат отдельно в `tests`;
- CLI написан на `click` и прописан в `[project.scripts]`;
- запуск типовых задач описан через Poe the Poet;
- документация собирается Sphinx, API генерируется из docstrings;
- CLI-документация генерируется через `sphinx-click`;
- проверки запускаются через `ruff`, type checker, pytest coverage;
- фикстуры pytest используются для подготовки файлов, временных папок и моков;
- побочные эффекты изолированы: CLI, файловая система и печать отделены от чистой логики;
- README содержит короткое назначение, badges и инструкции запуска;
- CI запускает быстрые проверки, тесты, документацию и сборку.

Для `edges` надо сохранить эту инженерную форму, но заменить доменную логику `pywc` на обработку изображений.

## Датасет

Основной датасет: **Berkeley Segmentation Dataset and Benchmark 500 (BSDS500)**.

Почему он подходит:

- это профильный набор для задач сегментации, boundaries и contour detection;
- содержит реальные natural images;
- содержит разделение на `train`, `val`, `test`;
- содержит human annotations, которые можно использовать позже для расширенной оценки качества контуров;
- датасет публично доступен для исследовательского и учебного применения.

Источники:

- официальный Berkeley resources page: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
- каталог BSDS на Berkeley: https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/

Важно: датасет не нужно коммитить в репозиторий. В Git хранятся только код, тестовые синтетические изображения и документация. Папки `data/`, `results/`, `reports/generated/` добавляются в `.gitignore`.

Резервный вариант, если официальный URL архива временно недоступен: CLI должен уметь работать с любой пользовательской папкой изображений, поэтому BSDS500 не должен быть жесткой зависимостью для запуска основной обработки.

## Алгоритмическая основа

Проект опирается на OpenCV:

- Sobel derivatives: https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html
- Laplacian derivatives: https://docs.opencv.org/4.x/d5/db5/tutorial_laplace_operator.html
- contours: https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

Обработка одного изображения:

1. Прочитать изображение через `cv2.imread`.
1. Проверить, что изображение не пустое.
1. Перевести BGR/RGB изображение в grayscale.
1. Опционально применить Gaussian blur для снижения шума.
1. Посчитать Sobel:
   - `grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=...)`;
   - `grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=...)`;
   - `magnitude = sqrt(grad_x ** 2 + grad_y ** 2)`;
   - нормализовать в `uint8` диапазон 0..255.
1. Посчитать Laplacian:
   - `lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=...)`;
   - взять абсолютные значения;
   - нормализовать в `uint8` диапазон 0..255.
1. Построить бинарную карту краев:
   - либо фиксированный `--threshold`;
   - либо Otsu threshold при `--threshold auto`.
1. Найти контуры через `cv2.findContours`.
1. Сохранить:
   - карту Sobel;
   - карту Laplacian;
   - бинарную карту краев;
   - изображение с наложенными контурами;
   - строку статистики в `summary.csv` и `summary.json`.

Контуры лучше строить не напрямую по цветному изображению, а по бинарной карте краев. Это делает результат воспроизводимым и тестируемым.

## Целевая структура репозитория

```text
edges/
  .github/
    workflows/
      ci.yml
  docs/
    _static/
      custom.css
    _templates/
    conf.py
    index.rst
    usage.md
    algorithms.md
    dataset.md
    performance.md
    cli.md
  reports/
    performance/
      performance.tex
      figures/
  src/
    edges/
      __init__.py
      __main__.py
      py.typed
      config.py
      console.py
      contours.py
      dataset.py
      images.py
      operators.py
      pipeline.py
      reporting.py
  tests/
    __init__.py
    conftest.py
    test_config.py
    test_console.py
    test_contours.py
    test_dataset.py
    test_images.py
    test_operators.py
    test_pipeline.py
    test_reporting.py
  .gitignore
  .pre-commit-config.yaml
  .python-version
  LICENSE
  README.md
  pyproject.toml
  uv.lock
  project.md
```

Папки, которые создаются во время работы и не коммитятся:

```text
data/
  raw/
    bsds500/
  processed/
results/
  bsds500-demo/
reports/
  generated/
```

## Модули пакета

### `src/edges/config.py`

Назначение: типизированные настройки обработки.

Основные dataclasses:

- `ProcessingConfig`
  - `method: Literal["sobel", "laplacian", "both"]`;
  - `blur_kernel: int`;
  - `sobel_kernel: int`;
  - `laplacian_kernel: int`;
  - `threshold: int | Literal["auto"]`;
  - `recursive: bool`;
  - `extensions: tuple[str, ...]`;
  - `draw_contours: bool`;
- `ImageResult`
  - `source_path: Path`;
  - `sobel_path: Path | None`;
  - `laplacian_path: Path | None`;
  - `binary_path: Path | None`;
  - `contours_path: Path | None`;
  - `width: int`;
  - `height: int`;
  - `edge_pixel_ratio: float`;
  - `contour_count: int`;
  - `largest_contour_area: float`;
  - `processing_ms: float`.

Стиль как в эталоне: `@dataclass(slots=True, kw_only=True)`, валидация в `__post_init__`, понятные Google-style docstrings.

### `src/edges/images.py`

Назначение: файловая система и чтение/запись изображений.

Функции:

- `iter_image_paths(input_dir: Path, extensions: Iterable[str], recursive: bool) -> Iterator[Path]`;
- `read_image(path: Path) -> NDArray[np.uint8]`;
- `save_grayscale(path: Path, image: NDArray[np.uint8]) -> None`;
- `save_color(path: Path, image: NDArray[np.uint8]) -> None`;
- `make_output_path(input_dir: Path, output_dir: Path, image_path: Path, suffix: str) -> Path`.

Правила:

- использовать `pathlib`, не собирать пути строковой конкатенацией;
- сохранять относительную структуру входной папки, чтобы не было конфликтов имен;
- если файл нельзя прочитать как изображение, выбрасывать доменное исключение `ImageReadError`.

### `src/edges/operators.py`

Назначение: чистые функции обработки массивов без файловой системы.

Функции:

- `to_grayscale(image: NDArray[np.uint8]) -> NDArray[np.uint8]`;
- `apply_blur(gray: NDArray[np.uint8], kernel_size: int) -> NDArray[np.uint8]`;
- `sobel_edges(gray: NDArray[np.uint8], kernel_size: int) -> NDArray[np.uint8]`;
- `laplacian_edges(gray: NDArray[np.uint8], kernel_size: int) -> NDArray[np.uint8]`;
- `normalize_to_uint8(values: NDArray[np.floating]) -> NDArray[np.uint8]`;
- `threshold_edges(edge_map: NDArray[np.uint8], threshold: int | Literal["auto"]) -> NDArray[np.uint8]`.

Этот модуль должен быть максимально легко тестируемым: вход - массив, выход - массив.

### `src/edges/contours.py`

Назначение: поиск и отрисовка контуров.

Функции:

- `find_contours(binary: NDArray[np.uint8], mode: int = cv2.RETR_EXTERNAL) -> list[NDArray[np.int32]]`;
- `draw_contours(image: NDArray[np.uint8], contours: Sequence[NDArray[np.int32]]) -> NDArray[np.uint8]`;
- `contour_stats(contours: Sequence[NDArray[np.int32]]) -> ContourStats`.

`ContourStats`:

- `count`;
- `largest_area`;
- `mean_area`;
- `mean_perimeter`.

### `src/edges/pipeline.py`

Назначение: сценарий обработки папки.

Функции:

- `process_image(image_path: Path, input_dir: Path, output_dir: Path, config: ProcessingConfig) -> ImageResult`;
- `process_directory(input_dir: Path, output_dir: Path, config: ProcessingConfig) -> list[ImageResult]`.

Здесь соединяются IO, операторы, контуры и отчеты. В CLI должна быть только подготовка аргументов и вызов `process_directory`.

### `src/edges/dataset.py`

Назначение: загрузка и подготовка открытого датасета.

Функции:

- `download_bsds500(target_dir: Path) -> Path`;
- `find_bsds_images(dataset_dir: Path, split: Literal["train", "val", "test", "all"]) -> list[Path]`;
- `copy_sample(source_dir: Path, target_dir: Path, limit: int) -> list[Path]`.

Реализация загрузки:

- использовать `pooch` или аналогичный downloader;
- проверять, что архив распакован в ожидаемую структуру;
- не требовать датасет для unit-тестов;
- в тестах мокать скачивание и распаковку.

### `src/edges/reporting.py`

Назначение: сохранение машинно-читаемой сводки и графиков.

Функции:

- `write_summary_csv(results: Sequence[ImageResult], path: Path) -> None`;
- `write_summary_json(results: Sequence[ImageResult], path: Path) -> None`;
- `plot_processing_times(results: Sequence[ImageResult], path: Path) -> None`;
- `plot_contour_counts(results: Sequence[ImageResult], path: Path) -> None`.

CSV/JSON нужны для проверки результата без ручного просмотра картинок. Графики нужны для документации и отчета о производительности.

### `src/edges/console.py`

Назначение: `click` CLI.

Команды:

```text
edges process INPUT_DIR OUTPUT_DIR
edges dataset download TARGET_DIR
edges dataset sample DATASET_DIR TARGET_DIR
edges benchmark INPUT_DIR OUTPUT_DIR
```

Основная команда:

```text
edges process data/raw/bsds500/images/test results/bsds500-test \
  --method both \
  --threshold auto \
  --blur-kernel 3 \
  --sobel-kernel 3 \
  --laplacian-kernel 3 \
  --recursive \
  --contours
```

Опции `process`:

- `--method [sobel|laplacian|both]`;
- `--threshold INTEGER|auto`;
- `--blur-kernel INTEGER`;
- `--sobel-kernel INTEGER`;
- `--laplacian-kernel INTEGER`;
- `--recursive / --no-recursive`;
- `--contours / --no-contours`;
- `--extension TEXT`, multiple option;
- `--limit INTEGER`, для быстрых демонстраций и профилирования;
- `--overwrite / --no-overwrite`;
- `--verbose`.

CLI должен возвращать ненулевой exit code при ошибках входной папки, неверных параметрах kernel size или невозможности записать результат.

## Формат результатов

Для входа:

```text
data/raw/bsds500/images/test/101085.jpg
```

выход:

```text
results/bsds500-test/
  sobel/101085_sobel.png
  laplacian/101085_laplacian.png
  binary/101085_binary.png
  contours/101085_contours.png
  summary.csv
  summary.json
  config.json
```

`summary.csv`:

```text
source_path,width,height,method,edge_pixel_ratio,contour_count,largest_contour_area,processing_ms
101085.jpg,481,321,both,0.0842,137,2324.5,18.7
```

`config.json` сохраняет параметры запуска, чтобы результат можно было воспроизвести.

## `pyproject.toml`

Проект оформляется как пакет.

Базовые зависимости:

```toml
[project]
name = "ippi-edges"
version = "0.1.0"
description = "Batch edge and contour detection for image folders."
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "click>=8.1",
  "numpy>=2",
  "opencv-python-headless>=4.10",
  "pooch>=1.8",
]

[project.scripts]
edges = "edges.console:main"
```

Группы зависимостей:

```toml
[dependency-groups]
dev = [
  "mdformat",
  "mypy",
  "poethepoet",
  "pre-commit",
  "pyproject-fmt",
  "ruff",
  { include-group = "docs" },
  { include-group = "tests" },
]
tests = [
  "coverage",
  "pytest",
  "pytest-cov",
  "pytest-mock",
  "pytest-randomly",
  "pytest-sugar",
  "pytest-xdist",
]
docs = [
  "furo",
  "myst-parser",
  "sphinx",
  "sphinx-autoapi",
  "sphinx-click",
  "sphinx-design",
  "sphinx-pyproject",
]
report = [
  "matplotlib",
]
build = [
  "pyinstaller",
  "twine",
]
```

Ruff:

- `select = ["ALL"]`;
- игнорировать только правила, которые конфликтуют с formatter или не относятся к проекту;
- для тестов разрешить `S101` из-за обычных `assert`;
- для `docs` можно исключить часть проверок, как в эталоне.

Coverage:

- `source = ["src/edges"]`;
- `fail_under = 95`;
- целиться в 100%, если исключения можно объяснить.

Type checking:

- основной вариант: `mypy src tests`;
- если у `cv2` появятся проблемы с типами, разрешается точечный ignore только для OpenCV-границы, не для всего проекта.

## Poe run targets

Минимальный набор:

```toml
[tool.poe.tasks]
edges = { cmd = "edges --help", help = "Show CLI help" }
process-demo = { cmd = "edges process data/sample results/demo --method both --threshold auto --contours", help = "Run demo processing" }
test = { cmd = "pytest --cov", help = "Run tests with coverage" }
test-fast = { cmd = "pytest -m 'not slow'", help = "Run fast tests only" }
format_py = { cmd = "ruff format src tests", help = "Format Python code" }
lint_py = { cmd = "ruff check src tests --fix", help = "Lint and fix Python code" }
check_py = { sequence = [
  { cmd = "ruff format --check src tests" },
  { cmd = "ruff check src tests" },
] }
typecheck = { cmd = "mypy src tests", help = "Run static type checking" }
build_doc = { sequence = [
  { cmd = "python -c \"import shutil; shutil.rmtree('docs/_build', ignore_errors=True); shutil.rmtree('docs/autoapi', ignore_errors=True)\"" },
  { cmd = "python -m sphinx -b html docs docs/_build/html" },
] }
profile = { cmd = "python -m cProfile -o reports/generated/profile.pstats -m edges process data/sample results/profile --method both --threshold auto --contours", help = "Profile demo processing" }
build_package = { cmd = "python -m build", help = "Build wheel and sdist" }
build_standalone = { cmd = "pyinstaller --onefile --name edges src/edges/__main__.py", help = "Build standalone executable" }
```

Композитные задачи:

```toml
[tool.poe.tasks.format]
sequence = [
  { cmd = "pyproject-fmt pyproject.toml", ignore_fail = true },
  { cmd = "mdformat README.md docs/*.md" },
  { ref = "format_py" },
]

[tool.poe.tasks.lint]
sequence = [
  { ref = "check_py" },
  { ref = "typecheck" },
]

[tool.poe.tasks.ci]
sequence = [
  { ref = "check_py" },
  { ref = "typecheck" },
  { ref = "test-fast" },
  { ref = "build_doc" },
  { ref = "build_package" },
]
```

## Тестирование

Тесты должны проверять логику без ручного просмотра картинок.

### Фикстуры

`tests/conftest.py`:

- `synthetic_square_image`: черный фон и белый квадрат;
- `synthetic_circle_image`: черный фон и белый круг;
- `gradient_image`: плавный градиент для Sobel;
- `image_folder`: временная папка с несколькими изображениями;
- `runner`: `click.testing.CliRunner`;
- `mock_downloader`: мок для загрузки BSDS500.

### Unit-тесты

`test_operators.py`:

- grayscale сохраняет размер;
- Sobel на равномерном изображении дает нулевую карту;
- Sobel на изображении с резкой вертикальной границей дает ненулевые края;
- Laplacian на равномерном изображении дает нулевую карту;
- threshold `auto` возвращает бинарную карту из значений 0 и 255;
- нечетные kernel sizes принимаются, четные отклоняются.

`test_contours.py`:

- квадрат дает хотя бы один контур;
- пустая binary map дает ноль контуров;
- `contour_stats` считает count, area и perimeter;
- draw не меняет размер изображения.

`test_images.py`:

- итерация находит только поддерживаемые расширения;
- recursive и non-recursive режимы отличаются;
- output path сохраняет относительный путь;
- ошибка чтения изображения обрабатывается явно.

`test_pipeline.py`:

- `process_image` создает ожидаемые файлы;
- `process_directory` обрабатывает все изображения;
- limit ограничивает количество обработанных изображений;
- summary содержит строки по всем обработанным изображениям.

`test_console.py`:

- `edges --help` работает;
- `edges --version` работает;
- неверная входная папка дает ошибку;
- CLI правильно передает параметры в `process_directory`;
- dataset-команды мокают скачивание и не ходят в сеть.

### Интеграционные и slow-тесты

Отдельные тесты с реальным маленьким набором изображений можно пометить:

```python
@pytest.mark.slow
```

CI запускает быстрые тесты без `slow`, локально можно запускать все.

## Документация

Минимум:

- `README.md` с назначением, установкой, запуском, примерами;
- `LICENSE`;
- Sphinx API docs из docstrings.

Кастомные страницы:

- `docs/usage.md` - запуск CLI и примеры входа/выхода;
- `docs/algorithms.md` - краткое объяснение Sobel, Laplacian, threshold и contours;
- `docs/dataset.md` - откуда берется BSDS500 и как подготовить папку;
- `docs/performance.md` - результаты профилирования, графики, выводы;
- `docs/cli.md` - автогенерация CLI через `sphinx-click`.

`docs/cli.md`:

````md
# Command Line Interface

```{eval-rst}
.. click:: edges.console:main
  :prog: edges
  :show-nested:
```
````

Тема документации: `furo`, как в эталоне.

## Производительность

Минимальный отчет:

1. Взять 50-100 изображений BSDS500.
1. Запустить обработку в базовом режиме.
1. Собрать профиль через `cProfile` или `py-spy`.
1. Найти bottleneck:
   - чтение/запись файлов;
   - нормализация массивов;
   - поиск контуров;
   - повторный grayscale/blur для разных методов.
1. Сделать ускорение:
   - не читать изображение дважды;
   - считать grayscale и blur один раз;
   - использовать векторизованные OpenCV/Numpy операции;
   - опционально добавить parallel processing через `concurrent.futures.ProcessPoolExecutor`.
1. Сравнить время до/после.
1. Сгенерировать графики Python-кодом.
1. Вставить графики и описание в LaTeX-документ `reports/performance/performance.tex`.

Метрики:

- общее время обработки;
- среднее время на изображение;
- изображения в секунду;
- время по этапам: read, preprocess, Sobel, Laplacian, contours, write.

## CI/CD

GitHub Actions:

```yaml
name: ci

on:
  pull_request:
  push:
    branches: [main]

jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v8.0.0
      - run: uv sync --locked --all-groups
      - run: uv run poe ci
```

В CI не скачивать полный BSDS500. Для CI используются синтетические изображения и маленькие fixtures. Полный датасет нужен для локальной демонстрации, профилирования и документационных примеров.

Badges в README:

- tests;
- coverage;
- docs;
- TestPyPI version после публикации.

## Packaging и релизы

TestPyPI:

```text
python -m build
uv run twine upload --repository testpypi dist/*
```

Standalone:

```text
uv run poe build_standalone
```

Для standalone-сборки лучше использовать `opencv-python-headless`, потому что GUI-возможности OpenCV не нужны: проект сохраняет файлы, а не открывает окна.

Артефакты GitHub Release:

- wheel;
- sdist;
- standalone executable;
- архив исходного кода.

Датасет и результаты обработки в релиз не включать.

## План реализации

1. Инициализировать проект через `uv init --package`.
1. Настроить `pyproject.toml`, `.python-version`, `.gitignore`, `.pre-commit-config.yaml`.
1. Создать `src/edges` и пустой typed package.
1. Реализовать `config.py`, `operators.py`, `contours.py`.
1. Написать unit-тесты на синтетических изображениях.
1. Реализовать `images.py` и `pipeline.py`.
1. Добавить CLI в `console.py` и прописать `[project.scripts]`.
1. Добавить `dataset.py` для BSDS500.
1. Добавить CSV/JSON summary и графики.
1. Настроить Sphinx-документацию.
1. Настроить Poe tasks.
1. Настроить CI.
1. Провести профилирование и внести ускорение.
1. Подготовить README, LICENSE, performance report.
1. Собрать пакет, проверить установку и CLI.

## Критерии готовности

Проект можно считать готовым, если выполняется:

- `uv run edges --help` показывает CLI;
- `uv run edges process <input> <output> --method both --contours` создает изображения и summary;
- `uv run poe test` проходит и показывает coverage не ниже целевого;
- `uv run poe lint` проходит;
- `uv run poe typecheck` проходит;
- `uv run poe build_doc` собирает документацию;
- `uv run poe build_package` собирает wheel/sdist;
- README содержит инструкцию запуска;
- Sphinx содержит минимум две кастомные страницы с примерами;
- есть отчет по производительности с графиками;
- CI запускает быстрые проверки.

## Основные риски и решения

Риск: полный BSDS500 слишком большой для CI.

Решение: не скачивать датасет в CI, использовать synthetic fixtures.

Риск: результаты Sobel/Laplacian зависят от параметров kernel/blur/threshold.

Решение: параметры фиксируются в `ProcessingConfig` и сохраняются в `config.json`.

Риск: тесты по пикселям могут быть хрупкими.

Решение: точные pixel-level проверки использовать только на простых синтетических массивах; для реальных изображений проверять форму, dtype, наличие файлов, допустимый диапазон и ненулевые summary-метрики.

Риск: OpenCV плохо типизируется в некоторых местах.

Решение: держать OpenCV-вызовы в узких модулях `images.py`, `operators.py`, `contours.py`; типизировать публичные функции через `numpy.typing.NDArray`; если нужно, делать точечные suppressions.

Риск: контуры на шумных изображениях дают слишком много мелких объектов.

Решение: добавить параметры blur, threshold и минимальной площади контура `--min-contour-area`.

## Итоговая архитектурная идея

Главное разделение:

- `console.py` - только CLI и преобразование аргументов;
- `pipeline.py` - orchestration обработки папки;
- `images.py` - файловая система и кодеки изображений;
- `operators.py` - Sobel, Laplacian, threshold как чистые функции;
- `contours.py` - поиск, отрисовка и статистика контуров;
- `reporting.py` - CSV/JSON/графики;
- `dataset.py` - загрузка открытого датасета.

Такой дизайн хорошо ложится на требования курса: код тестируемый, CLI понятный, документация автогенерируется, профилирование можно делать на уровне pipeline, а IO и внешние зависимости легко мокать.
