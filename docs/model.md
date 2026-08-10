# Model Contract

## Identity
This is a multi-class tourism business classification model built using **PyTorch**. It is designed to analyze text provided by a user to classify a business into relevant tourism categories.

## Problem it Solves
It automates the categorization of tourism businesses based on their textual description, value proposition, and services. Since a business might offer services spanning multiple categories, it supports multi-label classification.

## Inputs
The model expects the following input text fields (which may be concatenated or processed together):
- **Business Description**: A general description of what the business does.
- **Unique Value Proposition**: What makes the business stand out.
- **List of Services**: Specific services or amenities provided.

## Outputs
The model outputs a score from `0.0` to `1.0` for each of the following 8 classes:
1. **Coastal & Island** (Tourism centered around the ocean, beaches, islands, or marine activities)
2. **Adventure & Nature** (Tourism centered around outdoor nature, exploration, physical activity, or adventure)
3. **Cultural & Heritage** (Tourism centered around history, culture, traditions, religion, art, or heritage)
4. **Theme Parks / Entertainment** (Purpose-built entertainment destinations focused on amusement, rides, games, or shows)
5. **Urban & City** (Tourism centered around cities, modern urban environments, shopping, and city entertainment)
6. **Culinary & Gastronomy** (Tourism centered around food, drinks, dining, local cuisine, or culinary experiences)
7. **Accommodation & Staycation** (Tourism where lodging/staying at the property is the primary tourism product)
8. **OUT_OF_SCOPE**: The business/destination does not meaningfully belong to any of the 7 categories.

*Multiple categories can be selected when their scores exceed a chosen threshold.*

## Interpreting Confidence
- The scores (0-1) indicate the model's confidence that the business belongs to a category.
- **High confidence ≠ High correctness**: A high score means the text strongly matches patterns for that category, but it could still be a false positive (e.g., a restaurant that mentions a "stay" might be misclassified as a hotel).
- The `OUT_OF_SCOPE` class helps filter out irrelevant queries.
