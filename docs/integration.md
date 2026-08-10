# Integration Contract

## Division of Responsibility
- **Model**: Converts input text (description, value prop, services) into an array/dictionary of 8 scores (7 categories + 1 OUT_OF_SCOPE).
- **Backend**: Validates the input before sending it to the model. Applies the threshold to the scores to determine the final selected categories. Handles API errors and rate limits.
- **Training**: Training does *not* happen in the backend. The backend only uses a frozen, exported version of the model for inference.

## Example Request
```json
{
  "business_description": "We are a boutique hotel located in the heart of the city.",
  "unique_value_proposition": "Luxury stay with an authentic local experience.",
  "list_of_services": ["Accommodation", "Breakfast", "Guided City Tours"]
}
```

## Example Response
```json
{
  "scores": {
    "Coastal & Island": 0.05,
    "Adventure & Nature": 0.10,
    "Cultural & Heritage": 0.02,
    "Theme Parks / Entertainment": 0.01,
    "Urban & City": 0.15,
    "Culinary & Gastronomy": 0.30,
    "Accommodation & Staycation": 0.95,
    "OUT_OF_SCOPE": 0.01
  }
}
```
*(The backend applies a threshold, e.g. `0.5`, resulting in `["Accommodation", "Travel Services"]`)*
