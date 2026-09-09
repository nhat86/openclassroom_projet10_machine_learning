# Data Directory

This directory contains the data files for the employee attrition prediction model.

## Data Files

Place your CSV files in this directory:

- `extrait_sirh.csv` - HR information (employee demographics, job details)
- `extrait_eval.csv` - Evaluation data (performance ratings, satisfaction scores)
- `extrait_sondage.csv` - Survey data (engagement, work-life balance, training)

## Important Notes

- **Do not commit data files to Git** - they are excluded by .gitignore
- Update file paths in `src/train.py` to match your actual file locations
- Ensure data files are in the correct format and structure
