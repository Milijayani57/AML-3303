## Step 1: Raw / Messy Code (Before Principles)

# Messy code – not modular, not reusable, hard to maintain
import random

numbers = [random.randint(1, 100) for _ in range(10)]
print("Generated numbers:", numbers)

# Calculate average
total = 0
for n in numbers:
    total += n
average = total / len(numbers)
print("Average:", average)

# Find max
max_num = numbers[0]
for n in numbers:
    if n > max_num:
        max_num = n
print("Max:", max_num)


# """
# 🔴 Problems:

# No functions (not modular).

# Can’t reuse logic elsewhere.

# Hard to extend (e.g., adding min/median).

# Not scalable (works only for small lists).

# No error handling (reliability issue).

# No comments/documentation.
# """

## Step 2: Refactored Code (With Principles)

# import random
# from typing import List

# def generate_numbers(count: int, lower: int = 1, upper: int = 100) -> List[int]:
#     """Generate a list of random integers."""
#     return [random.randint(lower, upper) for _ in range(count)]

# def calculate_average(numbers: List[int]) -> float:
#     """Return the average of a list of numbers."""
#     if not numbers:
#         raise ValueError("List of numbers cannot be empty")
#     return sum(numbers) / len(numbers)

# def find_max(numbers: List[int]) -> int:
#     """Return the maximum number from a list."""
#     if not numbers:
#         raise ValueError("List of numbers cannot be empty")
#     return max(numbers)

# if __name__ == "__main__":
#     # Example workflow (can be reused in other projects)
#     nums = generate_numbers(10)
#     print("Generated numbers:", nums)
#     print("Average:", calculate_average(nums))
#     print("Max:", find_max(nums))

# """
# ✅ Improvements:

# Modularity: Code broken into functions.

# Reusability: Functions can be used in any project.

# Maintainability: Easy to add min/median later.

# Scalability: Can handle larger datasets (just change count).

# Reliability & Quality: Error handling included.

# Security & Trust: Checks against empty input.

# Collaboration: Docstrings/comments make it understandable for teams.
# """

### Classroom Activity 

## Step 1: Raw / Messy Pandas Code

import pandas as pd

# Load CSV
# df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# # Print average sepal length
# avg = df['sepal_length'].mean()
# print("Average sepal length:", avg)

# # Print max petal width
# mx = df['petal_width'].max()
# print("Max petal width:", mx)

# # Filter rows where species is setosa
# print(df[df['species'] == 'setosa'].head())




# """
# 🔴 Problems:

# All logic in one block → not modular.

# Hard to reuse functions for other datasets.

# No error handling → breaks if column names change.

# Not scalable (imagine working on multiple CSVs).

# No documentation → not good for collaboration.
# ""

"""Iris Dataset Analysis - Clean and modular version"""

import pandas as pd


def load_data(url):
    """Load CSV from URL with error handling."""
    try:
        return pd.read_csv(url)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def get_column_stat(df, column, stat='mean'):
    """Calculate statistics for a column safely."""
    if column not in df.columns:
        print(f"Column '{column}' not found. Available: {list(df.columns)}")
        return None
    return df[column].mean() if stat == 'mean' else df[column].max()


def filter_species(df, species, n=5):
    """Filter by species and return top n rows."""
    return df[df['species'] == species].head(n)


# Main analysis
if __name__ == "__main__":
    url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    
    df = load_data(url)
    if df is not None:
        avg = get_column_stat(df, 'sepal_length', 'mean')
        mx = get_column_stat(df, 'petal_width', 'max')
        
        print(f"Average sepal length: {avg:.2f}")
        print(f"Max petal width: {mx:.2f}")
        print("\nSetosa flowers:")
        print(filter_species(df, 'setosa'))