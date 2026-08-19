# Dataset Description

The dataset used in this project is an **essay-scoring dataset derived from the Automated Student Assessment Prize (ASAP) data**. It contains student essays together with human-assigned scores and additional metadata describing the essay, writing task, and student-related information.
The dataset is publicly available and can be downloaded here. 
https://github.com/scrosseye/ASAP_2.0

The main training file used by the project is:

```text
ASAP_2_Final_github_train.csv
```

## Dataset Overview

| Property | Description |
|---|---|
| **Number of training samples** | 17,307 essays |
| **Number of columns** | 15 |
| **Main text column** | `full_text` |
| **Target column** | `score` |
| **Score range** | 1–6 |
| **Task type** | Automatic Essay Scoring |
| **Model input** | Student essay text |
| **Model output** | Predicted essay score |
| **Problem type** | Regression / ordinal scoring |

The project also contains:

```text
ASAP_2_Final_github_test.csv
```

with approximately **7,421 samples**. However, the current implementation primarily loads the training CSV and creates its own training, validation, and testing subsets.

---

## Dataset Purpose

The dataset is used to teach the BERT model the relationship between:

```text
Student Essay
      ↓
Human Assigned Score
```

For example, if the training data contains:

```text
Essay A → Human Score 2
Essay B → Human Score 4
Essay C → Human Score 6
```

BERT learns linguistic and contextual patterns that distinguish lower-scoring essays from higher-scoring essays.

After training, when a completely new essay is provided, the system attempts to estimate the score that the essay would receive.

---

## Dataset Structure

Each row represents **one essay**.

A simplified representation is:

```text
┌──────────┬───────┬──────────────────────┬─────────────┐
│ essay_id │ score │ full_text            │ prompt_name │
├──────────┼───────┼──────────────────────┼─────────────┤
│ Essay 1  │   2   │ Student essay text   │ Prompt A    │
│ Essay 2  │   4   │ Student essay text   │ Prompt A    │
│ Essay 3  │   6   │ Student essay text   │ Prompt A    │
└──────────┴───────┴──────────────────────┴─────────────┘
```

The most important relationship for model training is:

```text
full_text → BERT → score
```

---

## Dataset Columns

The dataset contains **15 columns**.

| Column | Description | Used directly by BERT? |
|---|---|---|
| `essay_id` | Unique identifier assigned to each essay | No |
| `score` | Human-assigned essay score; this is the target variable | **Yes, as target** |
| `full_text` | Complete student essay | **Yes, as input** |
| `set` | Dataset/set identifier used by preprocessing | Used for preprocessing |
| `pubpriv` | Public/private dataset metadata | No |
| `assignment` | Writing assignment associated with the essay | No |
| `prompt_name` | Name or description of the writing prompt | No |
| `economically_disadvantaged` | Dataset demographic metadata | No |
| `student_disability_status` | Student disability-status metadata | No |
| `ell_status` | English Language Learner status | No |
| `race_ethnicity` | Dataset demographic metadata | No |
| `gender` | Dataset demographic metadata | No |
| `grade_level` | Student grade level | No |
| `essay_word_count` | Number of words in the essay | No |
| `task` | Type/category of writing task | No |

A very important point is that **the current BERT architecture does not feed all 15 columns into the model**.

The main BERT input is:

```text
full_text
```

and the training target is:

```text
score
```

The remaining columns mainly provide metadata and contextual information about the dataset.

---

## Important Dataset Fields

### `essay_id`

`essay_id` uniquely identifies an essay.

Example:

```text
essay_id = 10001
```

Its purpose is mainly data management. It is **not used to determine the essay score**.

### `full_text`

This is the most important input field.

It contains the complete text written by the student.

Example:

```text
The program provides students with an opportunity to learn
about different communities. It can also help students develop
communication and responsibility.
```

The project creates another column called `essay` from `full_text` after cleaning:

```python
df['essay'] = df['full_text'].apply(
    self.clean_text
)
```

Therefore:

```text
full_text
    ↓
clean_text()
    ↓
essay
    ↓
BERT Tokenizer
    ↓
BERT
```

### `score`

`score` is the **target variable**.

It represents the human-assigned quality score for the essay.

In the supplied training data, scores range approximately from:

```text
1 → lowest score
2
3
4
5
6 → highest score
```

During training, BERT tries to learn:

```text
Essay text → Human score
```

---

## Score Normalization

The project does not directly train BERT to output:

```text
1, 2, 3, 4, 5 or 6
```

Instead, the scores are normalized to a range between **0 and 1** using `MinMaxScaler`.

The important code is:

```python
self.scaler = MinMaxScaler()
```

followed by:

```python
df_copy.loc[
    mask,
    'normalized_score'
] = self.scaler.fit_transform(
    df_copy.loc[
        mask,
        [score_column]
    ]
)
```

For a score range of **1–6**, normalization works as follows:

| Original Human Score | Normalized Training Score |
|---:|---:|
| 1 | 0.00 |
| 2 | 0.20 |
| 3 | 0.40 |
| 4 | 0.60 |
| 5 | 0.80 |
| 6 | 1.00 |

The normalization formula is:

```text
                   score - minimum score
Normalized Score = ─────────────────────
                   maximum - minimum
```

For example, if the human score is `4`:

```text
Normalized Score
= (4 - 1) / (6 - 1)
= 3 / 5
= 0.60
```

Therefore, BERT is trained using:

```text
Human Score 4
      ↓
Normalized Score 0.60
      ↓
Training Target
```

---

## Sample Dataset Table

The following table illustrates the complete structure of the dataset. It includes **all 15 columns** described above. The essay texts and values shown here are synthetic examples created for readability and to demonstrate the dataset format.

| `essay_id` | `score` | `full_text` | `set` | `pubpriv` | `assignment` | `prompt_name` | `economically_disadvantaged` | `student_disability_status` | `ell_status` | `race_ethnicity` | `gender` | `grade_level` | `essay_word_count` | `task` |
|---|---:|---|---|---|---|---|---|---|---|---|---|---:|---:|---|
| SAMPLE_001 | 1 | I think the program is useful. People can learn and help others. | train | public | Synthetic AES Assignment | Program Participation | No | No | No | Not Reported | Not Reported | 6 | 12 | Text dependent |
| SAMPLE_002 | 2 | I believe the program is helpful because students can travel, learn about other people and take responsibility. | train | public | Synthetic AES Assignment | Program Participation | No | No | No | Not Reported | Not Reported | 6 | 18 | Text dependent |
| SAMPLE_003 | 3 | Joining the program would be a good opportunity. Participants can help communities and learn from different places. | train | public | Synthetic AES Assignment | Program Participation | No | No | No | Not Reported | Not Reported | 6 | 19 | Text dependent |
| SAMPLE_004 | 4 | The program offers meaningful work and educational experiences. Participants can support people while developing communication and teamwork skills. | train | public | Synthetic AES Assignment | Program Participation | No | No | No | Not Reported | Not Reported | 6 | 20 | Text dependent |
| SAMPLE_005 | 5 | I strongly recommend the program because it combines service, learning and personal growth while giving participants real responsibilities. | train | public | Synthetic AES Assignment | Program Participation | No | No | No | Not Reported | Not Reported | 6 | 20 | Text dependent |
| SAMPLE_006 | 6 | The program is an exceptional opportunity for personal growth and service. Participants can contribute to communities while developing resilience, leadership and cultural awareness. | train | public | Synthetic AES Assignment | Program Participation | No | No | No | Not Reported | Not Reported | 6 | 24 | Text dependent |

The sample table now contains every column present in the dataset:

```text
1.  essay_id
2.  score
3.  full_text
4.  set
5.  pubpriv
6.  assignment
7.  prompt_name
8.  economically_disadvantaged
9.  student_disability_status
10. ell_status
11. race_ethnicity
12. gender
13. grade_level
14. essay_word_count
15. task
```

Although all 15 columns are available in the dataset, the current BERT scoring architecture primarily uses `full_text` as the model input and `score` as the prediction target. The `set` field is also used during preprocessing. The other fields provide metadata and contextual information about each essay.

> **Note:** The rows in this sample table are synthetic examples used only to illustrate the 15-column dataset structure. They should not be treated as actual student records from the ASAP dataset.

---

## Dataset Summary

The dataset provides the **supervised learning examples** required to fine-tune BERT for essay scoring. Each essay contains a human score that acts as the ground-truth target.

The essay text is cleaned and tokenized before being passed to BERT, while the human score is normalized to the `0–1` range. The model learns the relationship between the contextual representation of the essay and its human-assigned score.

The core dataset flow is:

```text
ASAP CSV Dataset
       ↓
full_text + score
       ↓
Text Cleaning
       ↓
Cleaned Essay
       │
       ├──────────────► BERT Tokenization
       │                       ↓
       │                  BERT Model
       │                       ↓
       │                 Predicted Score
       │
       └── score ─► Normalization ─► Training Target
```
