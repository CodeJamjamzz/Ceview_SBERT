"""
Preprocessing logic for business description, value proposition, and services.
"""

def preprocess_input(description: str, value_prop: str, services: list) -> str:
    # Example placeholder: concatenating all text
    services_str = ", ".join(services)
    return f"{description} {value_prop} {services_str}"
