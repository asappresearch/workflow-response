################
CONTEXT = "<|context|>"
CONTEXT_END = "<|endofcontext|>"
USER =  "<|user|>"
USER_END =  "<|endofuser|>"
SYSTEM = "<|system|>"
WORKFLOW=  "<|workflow|>"
WORKFLOW_END= "<|endofworkflow|>"
ACTION = "<|action|>"
ACTION_END = "<|endofaction|>"
RESPONSE=  "<|response|>"
RESPONSE_END= "<|endofresponse|>"
################
SPECIAL_TOKEN_SET =  [
    CONTEXT,
    CONTEXT_END,
    USER,
    SYSTEM,
    WORKFLOW,
    WORKFLOW_END,
    USER,
    USER_END,
    ACTION,
    ACTION_END,
    RESPONSE,
    RESPONSE_END]


PAD = "<_pad_>"
UNK = "<_unk_>"
SOS = "<_sos_>"
EOS = "<_eos_>"
CONVO_START = "<_soc_>"
CONVO_END = "<_eoc_>"
CUS_START = "<_cus_start_>"
CUS_END = "<_cus_end_>"
REP_START = "<_rep_start_>"
REP_END = "<_rep_end_>"
BOT_START = "<_bot_start_>"
BOT_END = "<_bot_end_>"
REP_ACTION_START = "<_rep_action_start_>"
REP_ACTION_END = "<_rep_action_end_>"
RESPONSE_PLACEHOLDER = "<_response_placeholder_>"

REWARD_ZERO = "<_reward_zero_>"
REWARD_ONE = "<_reward_one_>"
SPLIT_TOKEN = "<|endoftext|>"

conversation_control_tokens = [
    CONVO_START, CONVO_END, CUS_START, CUS_END,
    REP_START, REP_END, BOT_START, BOT_END,
    REP_ACTION_START, REP_ACTION_END,
    RESPONSE_PLACEHOLDER,
]

# template tokens
NUMERIC = "__NUMBER__"
REP_NAME = "__REP_NAME__"
CUS_NAME = "__CUS_NAME__"
DATE = "__DATE__"
NUMBER = "__NUMBER__"
TIME = "__TIME__"
MONEY = "__MONEY__"
CONFIRMATION_CODE = "__CONFIRMATION_CODE__"
DENOM = "__DENOM__"
AIRPORT = "__AIRPORT__"
AIRLINE = "__AIRLINE__"
SEAT = "__SEAT__"
DAY_OF_WEEK = "__DAY_OF_WEEK__"
FLIGHT_NUMBER = "__FLIGHT_NUMBER__"
NUMBER_4_DIGIT = "__NUMBER_4_DIGIT__"
NUMBER_5_DIGIT = "__NUMBER_5_DIGIT__"
NUMBER_6_to_12_DIGIT = "__NUMBER_6_to_12_DIGIT__"
HOTEL_CONFIRMATION_NUMBER = "__HOTEL_CONFIRMATION_NUMBER__"
TRIP_ACTIONS_BOOKING_ID = "__TRIP_ACTIONS_BOOKING_ID__"
LOYALTY_NUMBER = "__LOYALTY_NUMBER__"
EMAIL = "__EMAIL__"
PHONE = "__PHONE__"

sanitized_tokens = [
    NUMERIC, REP_NAME, CUS_NAME, DATE, NUMBER, TIME, MONEY,
    CONFIRMATION_CODE, DENOM, AIRPORT, AIRLINE, SEAT,
    DAY_OF_WEEK, FLIGHT_NUMBER, NUMBER_4_DIGIT, NUMBER_5_DIGIT, NUMBER_6_to_12_DIGIT,
    HOTEL_CONFIRMATION_NUMBER, TRIP_ACTIONS_BOOKING_ID, TRIP_ACTIONS_BOOKING_ID,
    LOYALTY_NUMBER, EMAIL, PHONE,
]