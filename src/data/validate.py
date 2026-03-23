"""
Data validation with Pandera.
Builds schemas dynamically from config/config.yaml.
"""

import logging

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check

logger = logging.getLogger(__name__)


def _build_schema(cfg: dict, is_train: bool = True) -> DataFrameSchema:
    """Build a Pandera schema from config validation rules."""
    val_cfg = cfg["validation"]
    columns = {}

    # ── Target column ──
    target = cfg["data"]["target"]
    if is_train:
        target_rule = val_cfg.get("target", {})
        checks = []
        if "ge" in target_rule:
            checks.append(Check.ge(target_rule["ge"]))
        if "le" in target_rule:
            checks.append(Check.le(target_rule["le"]))
        columns[target] = Column(
            dtype=target_rule.get("dtype", "float32"),
            checks=checks,
            nullable=target_rule.get("nullable", False),
        )

    # ── Store column ──
    store_rule = val_cfg.get("store", {})
    store_checks = []
    if "isin" in store_rule:
        store_checks.append(Check.isin(store_rule["isin"]))
    if "ge" in store_rule:
        store_checks.append(Check.ge(store_rule["ge"]))
    columns["store"] = Column(
        dtype=store_rule.get("dtype", "int8"),
        checks=store_checks,
        nullable=False,
    )

    # ── Item column ──
    item_rule = val_cfg.get("item", {})
    item_checks = []
    if "ge" in item_rule:
        item_checks.append(Check.ge(item_rule["ge"]))
    if "le" in item_rule:
        item_checks.append(Check.le(item_rule["le"]))
    columns["item"] = Column(
        dtype=item_rule.get("dtype", "int8"),
        checks=item_checks,
        nullable=False,
    )

    return DataFrameSchema(
        columns=columns,
        strict=False,     # allow extra columns (features)
        coerce=True,
    )


def validate(df: pd.DataFrame, cfg: dict, is_train: bool = True) -> pd.DataFrame:
    """
    Validate a DataFrame against the config-defined schema.
    Raises pandera.errors.SchemaError if validation fails.
    """
    schema = _build_schema(cfg, is_train=is_train)
    label = "train" if is_train else "test"

    logger.info("Validating %s data (%d rows, %d cols)...", label, len(df), len(df.columns))
    validated = schema.validate(df, lazy=True)
    logger.info("[PASS] %s data validation passed.", label)

    return validated
