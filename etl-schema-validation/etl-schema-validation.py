def validate_records(records, schema):
    answers = []

    type_map = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool
    }

    for i, record in enumerate(records):
        record_errors = []

        for column_schema in schema:
            column = column_schema.get("column")
            expected_type = column_schema.get("type")
            nullable = column_schema.get("nullable", True)
            min_val = column_schema.get("min")
            max_val = column_schema.get("max")

            if column not in record:
                record_errors.append(f"{column}: missing")
                continue

            value = record[column]

            if value is None:
                if not nullable:
                    record_errors.append(f"{column}: null")
                continue

            if expected_type == "int":
                if isinstance(value, bool) or not isinstance(value, int):
                    record_errors.append(
                        f"{column}: expected int, got {type(value).__name__}"
                    )
                    continue

            elif expected_type == "float":
                # Accept both int and float
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    record_errors.append(
                        f"{column}: expected float, got {type(value).__name__}"
                    )
                    continue

            elif expected_type == "str":
                if not isinstance(value, str):
                    record_errors.append(
                        f"{column}: expected str, got {type(value).__name__}"
                    )
                    continue

            elif expected_type == "bool":
                if not isinstance(value, bool):
                    record_errors.append(
                        f"{column}: expected bool, got {type(value).__name__}"
                    )
                    continue

            # Range check
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if min_val is not None and value < min_val:
                    record_errors.append(
                        f"{column}: out of range"
                    )
                if max_val is not None and value > max_val:
                    record_errors.append(
                        f"{column}: out of range"
                    )

        is_valid = len(record_errors) == 0
        answers.append((i, is_valid, record_errors))

    return answers