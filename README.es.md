# SciELO OCA Metrics

[English](README.md) | [Português](README.pt.md) | [Español](README.es.md)

---

## Español

Una biblioteca de Python y conjunto de herramientas CLI para la extracción y computación de indicadores bibliométricos para el Observatorio de Ciencia Abierta de SciELO.

### Estructura

- `oca_metrics/adapters`: Adaptadores para diferentes fuentes de datos (Parquet, Elasticsearch, OpenSearch).
- `oca_metrics/preparation`: Herramientas para la preparación de datos (extracción de OpenAlex, procesamiento de SciELO, integración).
- `oca_metrics/utils`: Funciones de utilidad (métricas, normalización).
### Pruebas (Testing)

La suite de pruebas utiliza `pytest` y cubre los módulos de normalización, métricas, carga de categorías, adaptadores y preparación de datos SciELO.

Para ejecutar las pruebas:
```bash
# Instale las dependencias de prueba
pip install .[test]

# Ejecute pytest
pytest
```

### Instalación

```bash
pip install .
```

### Preparación de Datos (CLI)

La biblioteca proporciona la herramienta `oca-prep` para preparar los datos antes de la computación de las métricas.

#### 1. Extracción de OpenAlex
Extrae métricas de snapshots JSONL comprimidos de OpenAlex a archivos Parquet.
```bash
oca-prep extract-oa --base-dir /ruta/a/snapshots --output-dir ./oa-parquet
```

#### 2. Procesamiento SciELO
Carga y elimina duplicados (merge) de documentos SciELO.
```bash
oca-prep prepare-scielo --input articles.jsonl --output-jsonl scielo_merged.jsonl --strategies doi pid title
```

#### 3. Integración y Generación de Parquet Fusionado
Cruza los datos de SciELO con OpenAlex y genera el conjunto de datos final `merged_data.parquet`.
```bash
oca-prep integrate --scielo-jsonl scielo_merged.jsonl --oa-parquet-dir ./oa-parquet --output-parquet ./merged_data.parquet
```

### Computación de Métricas (CLI)

La biblioteca proporciona una herramienta de línea de comandos para computar indicadores bibliométricos:

```bash
oca-metrics --parquet data.parquet --global-xlsx meta.xlsx --year 2024 --level field
```

Argumentos principales:
- `--parquet`: Ruta al archivo Parquet (obligatorio).
- `--global-xlsx`: Ruta al archivo Excel de metadatos globales.
- `--year`: Año específico para el procesamiento.
- `--start-year` / `--end-year`: Rango de años (por defecto el año actual).
- `--level`: Nivel de agregación (`domain`, `field`, `subfield`, `topic`).
- `--output-file`: Nombre del archivo CSV de salida.

### Cómo ejecutar para todos los años y todos los niveles

Para procesar todos los años, utilice los argumentos `--start-year` y `--end-year` para definir el rango deseado. Para procesar todos los niveles de agregación (`domain`, `field`, `subfield`, `topic`), ejecute el comando repetidamente, cambiando el valor de `--level` en cada ejecución.

Ejemplo (bash):
```bash
for level in domain field subfield topic; do
  oca-metrics --parquet data.parquet --global-xlsx meta.xlsx --start-year 2018 --end-year 2024 --level $level --output-file "metrics_${level}.csv"
done
```
Esto generará un CSV para cada nivel, cubriendo todos los años del rango.

- Si no se pasa `--year`, `--start-year` o `--end-year`, el valor predeterminado es el año actual.
- El argumento `--level` acepta solo un valor por ejecución.

### Adaptadores Soportados

- **Parquet**: Utiliza DuckDB para un procesamiento eficiente de archivos locales o remotos.
- **Elasticsearch**: (Esqueleto) Soporte planificado para índices ES.
- **OpenSearch**: (Esqueleto) Soporte planificado para índices OpenSearch.

### Archivo Excel de Metadatos

El archivo Excel de metadatos globales (`--global-xlsx`) se utiliza para enriquecer los datos bibliométricos con información de las revistas. Las columnas esperadas son:

| Grupo | Nombre de Columna | Descripción |
| :--- | :--- | :--- |
| **Info Revista** | `journal_title` | Título de la revista. |
| | `journal_id` | Identificador OpenAlex de la revista (ej: S123456789). |
| | `publisher_name` | Nombre de la editorial. |
| | `journal_issn` | ISSNs asociados a la revista. |
| | `country` | El país responsable de la revista. |
| **Info SciELO** | `is_scielo` | Booleano que indica si la revista pertenece a la red SciELO. |
| | `scielo_active_valid` | Estado de la revista en la red SciELO para el año dado. |
| | `scielo_collection_acronym` | Acrónimo de la colección SciELO. |
| **Tiempo** | `publication_year` | El año de referencia para los metadatos. |

La biblioteca normaliza el `OpenAlex ID` al formato de URL utilizado en los datos Parquet (ej: `https://openalex.org/S...`).

### Esquema del Parquet

El archivo Parquet de entrada sirve como fuente de los datos de publicación. Debe contener las siguientes columnas para permitir la computación de métricas:

| Grupo | Nombre de Columna | Descripción |
| :--- | :--- | :--- |
| **Info Trabajo** | `work_id` | Identificador único del trabajo (publicación). |
| | `publication_year` | Año de publicación. |
| | `language` | Idioma de la publicación. |
| | `doi` | DOI de la publicación. |
| | `is_merged` | Booleano que indica si el registro está fusionado. |
| | `oa_individual_works` | JSON con detalles de los trabajos individuales (si está fusionado). |
| | `all_work_ids` | Lista de todos los IDs de trabajos en OpenAlex cuando 'is_merged' es True. |
| **Info Fuente** | `source_id` | Identificador de la fuente (revista), típicamente una URL de OpenAlex. |
| | `source_issn_l` | ISSN-L de la revista. |
| | `scielo_collection` | Colección SciELO de la publicación. |
| | `scielo_pid_v2` | PID v2 SciELO de la publicación. |
| **Categorías** | `domain` | Categoría de dominio. |
| | `field` | Categoría de campo. |
| | `subfield` | Categoría de subcampo. |
| | `topic` | Categoría de tema. |
| | `topic_score` | Puntuación de relevancia del tema. |
| **Citas** | `citations_total` | Total de citas recibidas. |
| | `citations_{year}` | Citas recibidas en el año respectivo. |
| | `citations_window_{w}y` | Citas recibidas en una ventana de {w} años. |
| | `has_citation_window_{w}y` | Booleano que indica si tiene citas en la ventana. |

### Esquema del CSV de Salida

El archivo CSV resultante contiene los indicadores bibliométricos computados, organizados por grupo:

| Grupo | Nombre de Columna | Descripción |
| :--- | :--- | :--- |
| **Contexto** | `category_level` | Nivel de agregación (ej: field, subfield). |
| | `category_id` | Identificador de la categoría (domain, field, etc.). |
| | `publication_year` | Año de publicación. |
| **Info Revista** | `journal_id` | OpenAlex ID (URL) de la revista. |
| | `journal_issn` | ISSN-L de la revista. |
| | `journal_title` | Título de la revista. |
| | `country` | El país responsable de la revista. |
| | `publisher_name` | Nombre de la editorial. |
| | `scielo_collection_acronym` | Acrônimo de la colección SciELO. |
| | `scielo_network_country` | País de la red SciELO. |
| | `scielo_active_valid` | Estado de la revista en SciELO. |
| | `is_scielo` | Indicador si la revista está en SciELO. |
| **Métricas Categoría** | `category_publications_count` | Total de publicaciones en la categoría en el año. |
| | `category_citations_total` | Total de citas recibidas por la categoría. |
| | `category_citations_mean` | Promedio de citas por publicación en la categoría. |
| | `category_citations_total_window_{w}y` | Total de citas en la ventana de {w} años. |
| | `category_citations_mean_window_{w}y` | Promedio de citas en la ventana de {w} años. |
| **Métricas Revista** | `journal_publications_count` | Total de publicaciones de la revista en el año. |
| | `journal_citations_total` | Total de citas recibidas por la revista. |
| | `journal_citations_mean` | Promedio de citas por publicación de la revista. |
| | `journal_impact_normalized` | Impacto normalizado (Promedio Revista / Promedio Categoría). |
| | `citations_window_{w}y` | Total de citas recibidas en la ventana de {w} años. |
| | `citations_window_{w}y_works` | Número de trabajos con al menos 1 cita en la ventana. |
| | `journal_citations_mean_window_{w}y` | Promedio de citas en la ventana de {w} años. |
| | `journal_impact_normalized_window_{w}y` | Impacto normalizado en la ventana de {w} años. |
| **Métricas Percentil** | `top_{pct}pct_all_time_citations_threshold` | Umbral de citas para el top {pct}% (todo el tiempo). |
| | `top_{pct}pct_all_time_publications_count` | Número de publicaciones en el top {pct}% (todo el tiempo). |
| | `top_{pct}pct_all_time_publications_share_pct` | Porcentaje de publicaciones en el top {pct}% (todo el tiempo). |
| | `top_{pct}pct_window_{w}y_citations_threshold` | Umbral de citas para el top {pct}% en ventana de {w} años. |
| | `top_{pct}pct_window_{w}y_publications_count` | Número de publicaciones en el top {pct}% en ventana de {w} años. |
| | `top_{pct}pct_window_{w}y_publications_share_pct` | Porcentaje de publicaciones en el top {pct}% en ventana de {w} años. |

> **Nota**: `{w}` representa el tamaño de la ventana (ej: 2, 3, 5) y `{pct}` representa el percentil (ej: 1, 5, 10, 50).

---

## Cómo funciona la fusión de documentos SciELO y OpenAlex

El proceso de fusión ocurre en varias etapas y puede personalizarse mediante estrategias de fusión (ej: `--strategies doi pid title`):

1. **Fusión SciELO-SciELO**:
   - **doi**: Los artículos se agrupan si comparten DOI (principal o por idioma) y títulos coincidentes.
   - **pid**: Los artículos se agrupan si comparten PIDv2, año de publicación, revista (por ISSN o título) y títulos coincidentes.
   - **title**: Los artículos se agrupan si comparten título (no genérico), año de publicación y revista (por ISSN o título).

2. **Vinculación SciELO-OpenAlex**:
   - Todos los DOIs de cada artículo SciELO fusionado se utilizan para buscar coincidencias en OpenAlex.
   - Cuando varios registros de OpenAlex coinciden con un SciELO, sus métricas se consolidan.

3. **Consolidación de métricas OpenAlex**:
   - Para cada artículo SciELO, todos los trabajos de OpenAlex encontrados tienen sus métricas agregadas.
   - Se preservan las métricas individuales de cada trabajo de OpenAlex y se computan los totales globales.

No hay una fusión explícita entre trabajos de OpenAlex; todos los trabajos de OpenAlex que coinciden con un SciELO se consolidan bajo ese artículo, eliminando duplicados cuando sea necesario.

Este proceso garantiza que cada artículo esté representado de forma única, con todos los metadatos y métricas relevantes consolidados de las fuentes SciELO y OpenAlex.

### Artículos multilingües y cálculo de métricas

Un solo artículo publicado en varios idiomas (por ejemplo, tres versiones) está representado en OpenAlex como tres documentos separados—uno por cada versión. En SciELO, se consideran un solo artículo. Esta distinción afecta el cálculo de métricas: contar cada documento de OpenAlex por separado inflaría el número de artículos publicados.

Para evitar esto, el proceso de fusión consolida todas las versiones y sus citas en un solo artículo. Así, la contribución total del artículo se calcula correctamente, reflejando todas las versiones y citas sin duplicidad.

---

## Clasificación de categorías y matemáticas de las métricas

Los artículos se clasifican en cuatro categorías jerárquicas: **domain**, **field**, **subfield** y **topic**. Todas las métricas bibliométricas se calculan y normalizan dentro de cada categoría y año de publicación. Esto permite comparar revistas de diferentes áreas de manera justa, ya que cada revista se evalúa en relación a su grupo de referencia.

### Normalización por categoría y año

Para cada categoría $c$ y año $y$, calculamos:

- Total de publicaciones: $N_{c,y}$
- Total de citas: $C_{c,y}$
- Promedio de citas por publicación: $\bar{C}_{c,y} = \frac{C_{c,y}}{N_{c,y}}$
- Total y promedio de citas en ventanas de tiempo $w$:
  - $C_{c,y}^{(w)}$: total de citas en la ventana $w$
  - $\bar{C}_{c,y}^{(w)} = \frac{C_{c,y}^{(w)}}{N_{c,y}}$

### Métricas de revistas

Para cada revista $j$ en la categoría $c$ y año $y$:

- Total de publicaciones: $N_{j,c,y}$
- Total de citas: $C_{j,c,y}$
- Promedio de citas por publicación: $\bar{C}_{j,c,y} = \frac{C_{j,c,y}}{N_{j,c,y}}$
- Citas en ventanas de tiempo $w$:
  - $C_{j,c,y}^{(w)}$: total de citas en la ventana $w$
  - $\bar{C}_{j,c,y}^{(w)} = \frac{C_{j,c,y}^{(w)}}{N_{j,c,y}}$

### Impacto normalizado

El impacto normalizado de la revista se calcula con protección para denominador cero:

$$
I_{j,c,y} =
\begin{cases}
0, & \text{si } \bar{C}_{c,y}=0 \\
\frac{\bar{C}_{j,c,y}}{\bar{C}_{c,y}}, & \text{en otro caso}
\end{cases}
$$

Y para ventanas de tiempo:

$$
I_{j,c,y}^{(w)} =
\begin{cases}
0, & \text{si } \bar{C}_{c,y}^{(w)}=0 \\
\frac{\bar{C}_{j,c,y}^{(w)}}{\bar{C}_{c,y}^{(w)}}, & \text{en otro caso}
\end{cases}
$$

### Percentiles y umbrales

Los umbrales se calculan para percentiles $p \in \{99,95,90,50\}$, que corresponden a top $q \in \{1,5,10,50\}$ donde $q=100-p$.

Para todas las citas en la categoría $c$ y año $y$:

$$
T_{c,y}^{(q)} = \left\lfloor Q_p\left(\mathcal{C}_{c,y}\right) \right\rfloor + 1
$$

Y para una ventana de citas $w$:

$$
T_{c,y}^{(q,w)} = \left\lfloor Q_p\left(\mathcal{C}_{c,y}^{(w)}\right) \right\rfloor + 1
$$

Los conteos de la revista en el top $q$% son:

$$
N_{j,c,y}^{(q)} = \sum_{i \in (j,c,y)} \mathbf{1}\!\left(C_{i,c,y} \ge T_{c,y}^{(q)}\right)
$$

$$
N_{j,c,y}^{(q,w)} = \sum_{i \in (j,c,y)} \mathbf{1}\!\left(C_{i,c,y}^{(w)} \ge T_{c,y}^{(q,w)}\right)
$$

Y el porcentaje de publicaciones es:

$$
S_{j,c,y}^{(q)} = \frac{N_{j,c,y}^{(q)}}{N_{j,c,y}} \times 100
$$

$$
S_{j,c,y}^{(q,w)} = \frac{N_{j,c,y}^{(q,w)}}{N_{j,c,y}} \times 100
$$

### Ejemplo práctico

Si una revista tiene 20 artículos en una categoría en 2024, con 100 citas totales:

- $\bar{C}_{j,c,2024} = \frac{100}{20} = 5$ citas por artículo
- Si el promedio de la categoría es 4, entonces $I_{j,c,2024} = \frac{5}{4} = 1.25$
- Si el umbral del top 5% es $T_{c,2024}^{(5)}=11$ y 2 artículos están por encima de ese umbral, entonces $S_{j,c,2024}^{(5)} = \frac{2}{20} \times 100 = 10\%$

Estas fórmulas permiten entender y comparar el desempeño de las revistas en cada área, ajustando por diferencias de tamaño e impacto.

---

## Estrategias de Fusión SciELO-OpenAlex y OpenAlex-OpenAlex

Esta biblioteca implementa dos procesos principales de fusión/consolidación:

**1. Fusión SciELO-OpenAlex**
- Para cada artículo SciELO, todos los DOIs (incluyendo variantes por idioma) se utilizan para buscar coincidencias en trabajos de OpenAlex en el Parquet.
- Si varios trabajos de OpenAlex coinciden con un solo artículo SciELO (por ejemplo, versiones multilingües), todos se agrupan bajo ese artículo SciELO.
- Para cada grupo, todas las métricas relevantes (citas, ventanas, etc.) se agregan (suman) para representar el impacto total del artículo, independientemente del idioma/versión.
- Los detalles individuales de cada trabajo de OpenAlex se preservan para referencia.
- Los campos de taxonomía (domain, field, subfield, topic) se consolidan a partir de todos los trabajos coincidentes.
- No se realiza fusión OpenAlex-OpenAlex en esta etapa; solo agrupamiento bajo artículos SciELO.

**2. Consolidación OpenAlex-OpenAlex**
- En el Parquet final, todos los trabajos de OpenAlex que coincidieron con un solo artículo SciELO se consolidan en un solo registro (el 'superviviente'), con todas las métricas agregadas.
- El campo 'all_work_ids' lista todos los IDs de OpenAlex que fueron fusionados.
- El campo 'is_merged' indica si el registro es resultado de fusionar múltiples trabajos de OpenAlex.
- El campo 'oa_individual_works' almacena los detalles de cada trabajo original de OpenAlex en formato JSON.
- Los trabajos de OpenAlex que no se asociaron a ningún artículo SciELO permanecen sin cambios, con 'is_merged' en False.
- Esto asegura que cada artículo esté representado de forma única, con todas las versiones y citas consolidadas, evitando doble conteo.

Este proceso de dos pasos garantiza una deduplicación robusta y un cálculo preciso de métricas para artículos publicados en varios idiomas o con variaciones de metadatos.

## Limitaciones y Cobertura

Solo los artículos SciELO que tienen taxonomía OpenAlex (domain, field, subfield, topic) se incluyen en las métricas por categoría. Es decir, solo los que tienen correspondencia en OpenAlex se cuentan en los denominadores de totales, promedios y percentiles. Los artículos SciELO sin match en OpenAlex aparecen en el Parquet final con citas en cero, pero se ignoran en las métricas por categoría porque no tienen taxonomía.

Aun así, es importante monitorear la cobertura de OpenAlex: áreas o revistas con baja correspondencia pueden tener métricas subestimadas o poco representativas. Se recomienda siempre revisar la proporción de artículos SciELO sin match en OpenAlex (por revista, año y categoría) antes de interpretar los resultados. Un informe de cobertura puede generarse en los logs para mayor transparencia.

### Cómo auditar la cobertura (artículos no emparejados)

Después de ejecutar el paso de integración (`oca-prep integrate`), se genera un archivo Parquet llamado `unmatched_scielo.parquet` en el directorio de salida. Este archivo contiene todos los artículos SciELO que no se emparejaron con ningún registro de OpenAlex. Puede analizar este archivo directamente para evaluar la cobertura e investigar los artículos no emparejados:

```python
import pandas as pd
unmatched = pd.read_parquet('unmatched_scielo.parquet')
print(unmatched.head())
print(f"Total no emparejados: {len(unmatched)}")
```

Este enfoque permite una auditoría transparente y detallada de las brechas de cobertura, sin necesidad de redirigir o analizar logs.

## Fuentes de Datos

- **OpenAlex**: Los datos se obtienen del snapshot de OpenAlex, específicamente del subconjunto SciELO. Ver: https://docs.openalex.org/download-all-data/openalex-snapshot
- **SciELO**: Los datos se obtienen de un volcado MongoDB de la base ArticleMeta (infraestructura interna de SciELO).
