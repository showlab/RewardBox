"""Capability taxonomy.  Also the data source of the survey's classification table."""
from enum import Enum


class RewardMode(str, Enum):
    PROGRESS = "progress"
    PREFERENCE = "preference"
    SUCCESS = "success"
    VALUE = "value"


class SignalType(str, Enum):
    TOKEN_LOGPROB = "token_logprob"      # TOPReward: log P(last token)
    GENERATED_TEXT = "generated_text"    # Robo-Dopamine, RoboReward
    VALUE_HEAD = "value_head"            # Robometer: custom progress/success head
    OPTION_PROBABILITY = "option_probability"  # Jev: probabilities over fixed answer options


class Normalize(str, Enum):
    NONE = "none"
    MINMAX = "minmax"
    ZSCORE = "zscore"
    RANK = "rank"


class ImplementationKind(str, Enum):
    """Evidence-level label.  It does not change the model algorithm; it only states
    where an adapter's faithfulness comes from.

    Set explicitly in every adapter's SPEC, never guessed from the class name.
    """
    OFFICIAL_ADAPTER = "official_adapter"            # TOPReward / Robo-Dopamine / Robometer
    MODEL_CARD_REFERENCE = "model_card_reference"     # RoboReward: no official code, built from the model card
    REWARDBOX_EXTENSION = "rewardbox_extension"       # reserved for pure RewardBox extensions
