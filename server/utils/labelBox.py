label_mapping = {0: "jewellery", 1: "face"}


def label_box(label_index: int):
    """
    Maps an integer label index to a string label.

    Args:
        label_index (int): The index to map (0 or 1).

    Returns:
        str: The corresponding label ('jewellery' or 'face').
    """

    return label_mapping.get(label_index, "Invalid label")
