"""
Employee Attendance & Analytics API

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience)
"""
import calendar
import os
import re
from contextlib import asynccontextmanager
from datetime import date as dt_date
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Annotated, Any, Dict, List, Optional

from bson import Decimal128, ObjectId, Timestamp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Path, Query
from pydantic import BaseModel, ConfigDict, Field, StrictInt, field_validator, model_validator
from pymongo import ASCENDING, DESCENDING, MongoClient, ReturnDocument
from pymongo.errors import DuplicateKeyError, PyMongoError

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "attendance_db")

client = MongoClient(MONGO_URI, tz_aware=True)
db = client[MONGO_DB]

IST = timezone(timedelta(hours=5, minutes=30))

MIN_EPOCH_MS = 100_000_000_000
MAX_EPOCH_MS = 4_102_444_800_000

EMP_CODE_RE = re.compile(r"^EMP\d{4,6}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
TIME_RE = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

EpochMillis = Annotated[StrictInt, Field(ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)]


# --------------------------------------------------------------------------- #
# Index Initialization & Lifespan
# --------------------------------------------------------------------------- #
def ensure_indexes() -> None:
    """Create all required indexes idempotently at startup."""
    # employees indexes
    db.employees.create_index([("emp_code", ASCENDING)], unique=True, name="idx_emp_code_unique")
    db.employees.create_index(
        [("department", ASCENDING), ("emp_code", ASCENDING)],
        name="idx_dept_emp_code",
    )
    db.employees.create_index(
        [("department", ASCENDING), ("joined_on", ASCENDING)],
        name="idx_dept_joined_on",
    )
    db.employees.create_index(
        [("joined_on", ASCENDING), ("department", ASCENDING)],
        name="idx_joined_on_dept",
    )

    # attendance_logs indexes
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("date", ASCENDING)],
        unique=True,
        name="idx_emp_date_unique",
    )
    db.attendance_logs.create_index(
        [("date", DESCENDING), ("emp_code", ASCENDING)],
        name="idx_date_desc_emp_asc",
    )
    db.attendance_logs.create_index(
        [("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)],
        name="idx_status_date_desc_emp_asc",
    )
    db.attendance_logs.create_index(
        [("emp_code", ASCENDING), ("punch_in", DESCENDING)],
        name="idx_emp_punch_in_desc",
    )
    db.attendance_logs.create_index(
        [("date", ASCENDING), ("late_minutes", ASCENDING), ("emp_code", ASCENDING)],
        name="idx_date_late_emp",
    )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        ensure_indexes()
    except PyMongoError:
        pass
    yield


app = FastAPI(
    title="Employee Attendance & Analytics API",
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Enums & Validation Helpers
# --------------------------------------------------------------------------- #
class Status(str, Enum):
    PRESENT = "PRESENT"
    ABSENT = "ABSENT"
    LEAVE = "LEAVE"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class PresenceStatus(str, Enum):
    PRESENT = "PRESENT"
    WFH = "WFH"
    ON_DUTY = "ON_DUTY"


class ExplainTarget(str, Enum):
    attendance_list = "attendance_list"
    employee_monthly = "employee_monthly"
    department_summary = "department_summary"
    late_leaderboard = "late_leaderboard"
    department_trend = "department_trend"


PRESENCE_VALUES = {Status.PRESENT.value, Status.WFH.value, Status.ON_DUTY.value}


def parse_iso_date(date_str: str, field_name: str = "date") -> dt_date:
    if not isinstance(date_str, str) or not DATE_RE.match(date_str):
        raise ValueError(f"Invalid {field_name}; expected YYYY-MM-DD")
    try:
        return dt_date.fromisoformat(date_str)
    except ValueError as exc:
        raise ValueError(f"Invalid calendar date for {field_name}") from exc


def validate_date_param(date_str: str, field_name: str = "date") -> dt_date:
    try:
        return parse_iso_date(date_str, field_name)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def validate_month_param(month_str: str) -> tuple[str, str, int, int]:
    if not isinstance(month_str, str) or not MONTH_RE.match(month_str):
        raise HTTPException(status_code=422, detail="Invalid month; expected YYYY-MM")
    year, month = map(int, month_str.split("-"))
    last_day = calendar.monthrange(year, month)[1]
    month_start = f"{year:04d}-{month:02d}-01"
    month_end = f"{year:04d}-{month:02d}-{last_day:02d}"
    return month_start, month_end, year, month


def now_truncated_utc() -> datetime:
    """Return current UTC datetime truncated to whole seconds (R1)."""
    sec = int(datetime.now(timezone.utc).timestamp())
    return datetime.fromtimestamp(sec, tz=timezone.utc)


def ms_to_utc_dt(ms: int) -> datetime:
    """Truncate epoch ms to whole seconds and return UTC-aware datetime (R1)."""
    truncated_sec = int(ms) // 1000
    return datetime.fromtimestamp(truncated_sec, tz=timezone.utc)


def dt_to_epoch_ms(dt_val: Optional[datetime]) -> Optional[int]:
    """Convert BSON datetime (naive or UTC-aware) to integer epoch milliseconds."""
    if dt_val is None:
        return None
    if dt_val.tzinfo is None:
        dt_val = dt_val.replace(tzinfo=timezone.utc)
    return int(dt_val.timestamp()) * 1000


def ensure_utc(dt_val: Optional[datetime]) -> Optional[datetime]:
    if dt_val is None:
        return None
    if dt_val.tzinfo is None:
        return dt_val.replace(tzinfo=timezone.utc)
    return dt_val.astimezone(timezone.utc)


def round_half_up(value: float | int | Decimal, decimals: int = 2) -> float:
    d = Decimal(str(value))
    target = Decimal("10") ** (-decimals)
    return float(d.quantize(target, rounding=ROUND_HALF_UP))


def mongo_round_half_up(expr: Any, decimals: int) -> Dict[str, Any]:
    """Build a MongoDB aggregation expression for exact ROUND_HALF_UP on non-negative numbers."""
    factor = 10**decimals
    return {
        "$toDouble": {
            "$divide": [
                {
                    "$floor": {
                        "$add": [
                            {"$multiply": [{"$toDecimal": expr}, factor]},
                            {"$toDecimal": "0.5"},
                        ]
                    }
                },
                factor,
            ]
        }
    }


# --------------------------------------------------------------------------- #
# Business Rule Helpers (R1 - R7)
# --------------------------------------------------------------------------- #
def compute_attendance_date(punch_in: datetime, shift_start: str, shift_end: str) -> str:
    """Compute IST attendance date per R1, handling overnight shifts."""
    ist_dt = ensure_utc(punch_in).astimezone(IST)
    eh, em = map(int, shift_end.split(":"))
    if shift_end <= shift_start and (ist_dt.hour, ist_dt.minute, ist_dt.second) < (eh, em, 0):
        return (ist_dt.date() - timedelta(days=1)).isoformat()
    return ist_dt.date().isoformat()


def compute_late_minutes(punch_in: datetime, shift_start: str, date_str: Optional[str] = None) -> int:
    """Compute late minutes per R2 (strictly > 10:00 grace period, floored to whole minutes)."""
    ist_punch = ensure_utc(punch_in).astimezone(IST)
    d_obj = dt_date.fromisoformat(date_str) if date_str else ist_punch.date()
    sh, sm = map(int, shift_start.split(":"))
    shift_start_dt = datetime(d_obj.year, d_obj.month, d_obj.day, sh, sm, 0, tzinfo=IST)
    diff_seconds = (ist_punch - shift_start_dt).total_seconds()
    if diff_seconds > 600:
        return int(diff_seconds // 60)
    return 0


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """Compute work hours per R4: (punch_out - punch_in) in seconds / 3600, 2 decimals half-up."""
    p_in = ensure_utc(punch_in)
    p_out = ensure_utc(punch_out)
    seconds = Decimal(int(round((p_out - p_in).total_seconds())))
    hours = (seconds / Decimal(3600)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(hours)


def compute_overtime(
    punch_out: datetime,
    shift_end: str,
    date_str: str,
    shift_start: Optional[str] = None,
) -> int:
    """Compute overtime minutes per R3 (whole minutes >= 30 after shift_end, handling overnight shifts)."""
    p_out = ensure_utc(punch_out).astimezone(IST)
    d_obj = dt_date.fromisoformat(date_str)
    eh, em = map(int, shift_end.split(":"))
    shift_end_dt = datetime(d_obj.year, d_obj.month, d_obj.day, eh, em, 0, tzinfo=IST)
    if shift_start is not None and shift_end <= shift_start:
        shift_end_dt += timedelta(days=1)
    elif shift_start is None and p_out.date() > d_obj and (eh, em) <= (12, 0):
        shift_end_dt += timedelta(days=1)
    diff_seconds = (p_out - shift_end_dt).total_seconds()
    mins = int(diff_seconds // 60)
    return mins if mins >= 30 else 0


def compute_working_days(month_start: str, month_end: str, joined_on: str) -> int:
    """Count Mon-Fri working days in the month on or after joined_on (R7)."""
    if joined_on > month_end:
        return 0
    start_str = max(month_start, joined_on)
    cur = dt_date.fromisoformat(start_str)
    end = dt_date.fromisoformat(month_end)
    count = 0
    while cur <= end:
        if cur.isoweekday() <= 5:
            count += 1
        cur += timedelta(days=1)
    return count


# --------------------------------------------------------------------------- #
# Serialization Helpers
# --------------------------------------------------------------------------- #
def serialize_employee(doc: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "emp_code": doc["emp_code"],
        "name": doc["name"],
        "email": doc["email"],
        "department": doc["department"],
        "shift_start": doc["shift_start"],
        "shift_end": doc["shift_end"],
        "joined_on": doc["joined_on"],
        "created_at": dt_to_epoch_ms(doc["created_at"]),
    }


def serialize_history_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    raw_changes = entry.get("changes") or {}
    serialized_changes: Dict[str, Dict[str, Any]] = {}
    for field, diff in raw_changes.items():
        f_val = diff.get("from")
        t_val = diff.get("to")
        if field in ("punch_in", "punch_out"):
            f_val = dt_to_epoch_ms(f_val) if isinstance(f_val, datetime) else f_val
            t_val = dt_to_epoch_ms(t_val) if isinstance(t_val, datetime) else t_val
        serialized_changes[field] = {"from": f_val, "to": t_val}
    return {
        "at": dt_to_epoch_ms(entry["at"]) if isinstance(entry.get("at"), datetime) else entry.get("at"),
        "by": entry["by"],
        "reason": entry["reason"],
        "changes": serialized_changes,
    }


def serialize_attendance(doc: Dict[str, Any]) -> Dict[str, Any]:
    raw_history = doc.get("history") or []
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_epoch_ms(doc.get("punch_in")),
        "punch_out": dt_to_epoch_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": int(doc.get("late_minutes") or 0),
        "overtime_minutes": int(doc.get("overtime_minutes") or 0),
        "half_day": bool(doc.get("half_day") or False),
        "history": [serialize_history_entry(h) for h in raw_history],
    }


def sanitize_bson(obj: Any) -> Any:
    """Recursively convert MongoDB BSON types in explain output to JSON-safe primitives."""
    if isinstance(obj, dict):
        return {str(k): sanitize_bson(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_bson(v) for v in obj]
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, Decimal128):
        return float(obj.to_decimal())
    if isinstance(obj, Timestamp):
        return {"t": obj.time, "i": obj.inc}
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.hex()
    return obj


# --------------------------------------------------------------------------- #
# Pydantic Models
# --------------------------------------------------------------------------- #
class EmployeeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(..., pattern=r"^EMP\d{4,6}$")
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(..., min_length=1, max_length=50)
    shift_start: str = Field(default="09:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    shift_end: str = Field(default="18:30", pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    joined_on: str

    @field_validator("joined_on")
    @classmethod
    def check_joined_on(cls, v: str) -> str:
        parse_iso_date(v, "joined_on")
        return v

    @model_validator(mode="after")
    def check_shifts_differ(self) -> "EmployeeIn":
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start and shift_end must differ")
        return self


class PunchInIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(..., min_length=1)
    punched_at: Optional[EpochMillis] = None
    status: PresenceStatus = PresenceStatus.PRESENT

    @model_validator(mode="after")
    def check_explicit_nulls(self) -> "PunchInIn":
        if "punched_at" in self.model_fields_set and self.punched_at is None:
            raise ValueError("punched_at cannot be null when provided")
        if "status" in self.model_fields_set and self.status is None:
            raise ValueError("status cannot be null")
        return self


class PunchOutIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    emp_code: str = Field(..., min_length=1)
    punched_at: Optional[EpochMillis] = None

    @model_validator(mode="after")
    def check_explicit_nulls(self) -> "PunchOutIn":
        if "punched_at" in self.model_fields_set and self.punched_at is None:
            raise ValueError("punched_at cannot be null when provided")
        return self


class RegularizeIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    status: Optional[Status] = None
    punch_in: Optional[EpochMillis] = None
    punch_out: Optional[EpochMillis] = None
    reason: str = Field(..., min_length=5, max_length=200)
    regularized_by: str = Field(..., min_length=1, max_length=50)

    @model_validator(mode="after")
    def check_explicit_nulls(self) -> "RegularizeIn":
        for f in ("status", "punch_in", "punch_out"):
            if f in self.model_fields_set and getattr(self, f) is None:
                raise ValueError(f"{f} cannot be null when provided")
        return self


# --------------------------------------------------------------------------- #
# Query & Aggregation Pipeline Builders (Shared by Endpoints & Explain)
# --------------------------------------------------------------------------- #
def build_attendance_list_filter(
    emp_code: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    status: Optional[Status],
) -> Dict[str, Any]:
    if date_from is not None:
        validate_date_param(date_from, "date_from")
    if date_to is not None:
        validate_date_param(date_to, "date_to")
    if date_from is not None and date_to is not None and date_from > date_to:
        raise HTTPException(status_code=422, detail="date_from must not be after date_to")

    q: Dict[str, Any] = {}
    if emp_code:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status is not None:
        q["status"] = status.value if isinstance(status, Status) else status
    return q


def build_employee_monthly_pipeline(
    emp_code: str,
    month: str,
    working_days: int,
) -> List[Dict[str, Any]]:
    month_start, month_end, _, _ = validate_month_param(month)
    is_weekday_expr = {
        "$lte": [
            {
                "$isoDayOfWeek": {
                    "$dateFromString": {
                        "dateString": "$date",
                        "format": "%Y-%m-%d",
                    }
                }
            },
            5,
        ]
    }
    return [
        {
            "$match": {
                "emp_code": emp_code,
                "date": {"$gte": month_start, "$lte": month_end},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "raw_present_days": {
                    "$sum": {
                        "$cond": [
                            {
                                "$and": [
                                    {"$in": ["$status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    is_weekday_expr,
                                ]
                            },
                            {"$cond": [{"$eq": ["$half_day", True]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "leave_days": {
                    "$sum": {"$cond": [{"$eq": ["$status", "LEAVE"]}, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$late_minutes", 0]}
                },
                "total_overtime_minutes": {
                    "$sum": {"$ifNull": ["$overtime_minutes", 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "emp_code": {"$literal": emp_code},
                "month": {"$literal": month},
                "working_days": {"$literal": working_days},
                "present_days": mongo_round_half_up("$raw_present_days", 2),
                "leave_days": 1,
                "late_count": 1,
                "total_late_minutes": 1,
                "total_overtime_minutes": 1,
                "attendance_pct": (
                    mongo_round_half_up(
                        {
                            "$divide": [
                                {"$multiply": [{"$toDecimal": "$raw_present_days"}, 100]},
                                working_days,
                            ]
                        },
                        2,
                    )
                    if working_days > 0
                    else {"$literal": None}
                ),
            }
        },
    ]


def build_department_summary_pipeline(
    month: str,
    department: Optional[str] = None,
) -> List[Dict[str, Any]]:
    month_start, month_end, _, _ = validate_month_param(month)
    emp_match: Dict[str, Any] = {"joined_on": {"$lte": month_end}}
    if department is not None:
        emp_match["department"] = department

    is_present_log = {"$in": ["$logs.status", ["PRESENT", "WFH", "ON_DUTY"]]}
    is_weekday_log = {
        "$lte": [
            {
                "$isoDayOfWeek": {
                    "$dateFromString": {
                        "dateString": "$logs.date",
                        "format": "%Y-%m-%d",
                    }
                }
            },
            5,
        ]
    }
    has_work_hours = {
        "$and": [
            is_present_log,
            {"$ne": [{"$ifNull": ["$logs.work_hours", None]}, None]},
        ]
    }

    return [
        {"$match": emp_match},
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"ec": "$emp_code"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$emp_code", "$$ec"]},
                                    {"$gte": ["$date", month_start]},
                                    {"$lte": ["$date", month_end]},
                                ]
                            }
                        }
                    }
                ],
                "as": "logs",
            }
        },
        {"$unwind": {"path": "$logs", "preserveNullAndEmptyArrays": True}},
        {
            "$group": {
                "_id": "$department",
                "emp_set": {"$addToSet": "$emp_code"},
                "raw_present_days": {
                    "$sum": {
                        "$cond": [
                            {"$and": [is_present_log, is_weekday_log]},
                            {"$cond": [{"$eq": ["$logs.half_day", True]}, 0.5, 1.0]},
                            0.0,
                        ]
                    }
                },
                "work_hours_sum": {
                    "$sum": {
                        "$cond": [
                            has_work_hours,
                            {"$toDecimal": "$logs.work_hours"},
                            {"$toDecimal": 0},
                        ]
                    }
                },
                "work_hours_count": {
                    "$sum": {"$cond": [has_work_hours, 1, 0]}
                },
                "late_count": {
                    "$sum": {"$cond": [{"$gt": ["$logs.late_minutes", 0]}, 1, 0]}
                },
                "total_late_minutes": {
                    "$sum": {"$ifNull": ["$logs.late_minutes", 0]}
                },
                "leave_count": {
                    "$sum": {"$cond": [{"$eq": ["$logs.status", "LEAVE"]}, 1, 0]}
                },
                "on_duty_count": {
                    "$sum": {"$cond": [{"$eq": ["$logs.status", "ON_DUTY"]}, 1, 0]}
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "department": "$_id",
                "headcount": {"$size": "$emp_set"},
                "present_days": mongo_round_half_up("$raw_present_days", 2),
                "avg_work_hours": {
                    "$cond": [
                        {"$gt": ["$work_hours_count", 0]},
                        mongo_round_half_up(
                            {"$divide": ["$work_hours_sum", "$work_hours_count"]},
                            2,
                        ),
                        None,
                    ]
                },
                "late_count": 1,
                "total_late_minutes": 1,
                "leave_count": 1,
                "on_duty_count": 1,
            }
        },
        {"$sort": {"department": 1}},
    ]


def build_late_leaderboard_pipeline(
    month: str,
    limit: int = 10,
    department: Optional[str] = None,
) -> List[Dict[str, Any]]:
    month_start, month_end, _, _ = validate_month_param(month)
    pipeline: List[Dict[str, Any]] = [
        {
            "$match": {
                "date": {"$gte": month_start, "$lte": month_end},
                "late_minutes": {"$gt": 0},
            }
        },
        {
            "$group": {
                "_id": "$emp_code",
                "total_late_minutes": {"$sum": "$late_minutes"},
                "late_count": {"$sum": 1},
            }
        },
        {
            "$lookup": {
                "from": "employees",
                "localField": "_id",
                "foreignField": "emp_code",
                "as": "emp",
            }
        },
        {"$unwind": "$emp"},
    ]
    if department is not None:
        pipeline.append({"$match": {"emp.department": department}})

    pipeline.extend(
        [
            {
                "$setWindowFields": {
                    "sortBy": {"total_late_minutes": -1},
                    "output": {"rank": {"$rank": {}}},
                }
            },
            {"$match": {"rank": {"$lte": limit}}},
            {"$sort": {"total_late_minutes": -1, "_id": 1}},
            {
                "$project": {
                    "_id": 0,
                    "rank": 1,
                    "emp_code": "$_id",
                    "name": "$emp.name",
                    "department": "$emp.department",
                    "total_late_minutes": 1,
                    "late_count": 1,
                }
            },
        ]
    )
    return pipeline


def build_department_trend_pipeline(
    department: str,
    from_date: str,
    to_date: str,
) -> List[Dict[str, Any]]:
    d_from = validate_date_param(from_date, "from")
    d_to = validate_date_param(to_date, "to")
    if d_to < d_from:
        raise HTTPException(status_code=422, detail="'to' must not be earlier than 'from'")
    if (d_to - d_from).days + 1 > 92:
        raise HTTPException(status_code=422, detail="Date range must not exceed 92 days")

    start_dt = datetime(d_from.year, d_from.month, d_from.day, tzinfo=timezone.utc)
    end_exclusive_dt = datetime(d_to.year, d_to.month, d_to.day, tzinfo=timezone.utc) + timedelta(days=1)

    return [
        {"$match": {"department": department}},
        {
            "$group": {
                "_id": None,
                "emp_codes": {"$push": "$emp_code"},
                "join_dates": {"$push": "$joined_on"},
            }
        },
        {"$addFields": {"dt": start_dt}},
        {
            "$densify": {
                "field": "dt",
                "range": {
                    "step": 1,
                    "unit": "day",
                    "bounds": [start_dt, end_exclusive_dt],
                },
            }
        },
        {
            "$fill": {
                "sortBy": {"dt": 1},
                "output": {
                    "emp_codes": {"method": "locf"},
                    "join_dates": {"method": "locf"},
                },
            }
        },
        {
            "$addFields": {
                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$dt"}},
                "is_working_day": {"$lte": [{"$isoDayOfWeek": "$dt"}, 5]},
            }
        },
        {
            "$addFields": {
                "headcount": {
                    "$size": {
                        "$filter": {
                            "input": "$join_dates",
                            "as": "jd",
                            "cond": {"$lte": ["$$jd", "$date"]},
                        }
                    }
                }
            }
        },
        {
            "$lookup": {
                "from": "attendance_logs",
                "let": {"d": "$date", "codes": "$emp_codes"},
                "pipeline": [
                    {
                        "$match": {
                            "$expr": {
                                "$and": [
                                    {"$eq": ["$date", "$$d"]},
                                    {"$in": ["$emp_code", "$$codes"]},
                                ]
                            }
                        }
                    }
                ],
                "as": "day_logs",
            }
        },
        {
            "$addFields": {
                "raw_present_count": {
                    "$sum": {
                        "$map": {
                            "input": "$day_logs",
                            "as": "log",
                            "in": {
                                "$cond": [
                                    {"$in": ["$$log.status", ["PRESENT", "WFH", "ON_DUTY"]]},
                                    {"$cond": [{"$eq": ["$$log.half_day", True]}, 0.5, 1.0]},
                                    0.0,
                                ]
                            },
                        }
                    }
                },
                "late_count": {
                    "$size": {
                        "$filter": {
                            "input": "$day_logs",
                            "as": "log",
                            "cond": {"$gt": ["$$log.late_minutes", 0]},
                        }
                    }
                },
            }
        },
        {
            "$addFields": {
                "present_count": mongo_round_half_up("$raw_present_count", 2),
                "attendance_rate": {
                    "$cond": [
                        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
                        mongo_round_half_up(
                            {"$divide": [{"$toDecimal": "$raw_present_count"}, "$headcount"]},
                            4,
                        ),
                        None,
                    ]
                },
            }
        },
        {
            "$addFields": {
                "rate_decimal": {
                    "$cond": [
                        {"$eq": ["$attendance_rate", None]},
                        None,
                        {"$toDecimal": "$attendance_rate"},
                    ]
                }
            }
        },
        {
            "$setWindowFields": {
                "sortBy": {"date": 1},
                "output": {
                    "raw_moving_avg_7d": {
                        "$avg": "$rate_decimal",
                        "window": {"documents": [-6, 0]},
                    }
                },
            }
        },
        {
            "$project": {
                "_id": 0,
                "date": 1,
                "is_working_day": 1,
                "headcount": 1,
                "present_count": 1,
                "late_count": 1,
                "attendance_rate": 1,
                "moving_avg_7d": {
                    "$cond": [
                        {"$eq": ["$raw_moving_avg_7d", None]},
                        None,
                        mongo_round_half_up("$raw_moving_avg_7d", 4),
                    ]
                },
            }
        },
    ]


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
        return {"status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = now_truncated_utc()
    try:
        db.employees.insert_one(doc)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="emp_code already exists") from exc
    return serialize_employee(doc)


@app.get("/employees")
def list_employees(
    department: Optional[str] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q: Dict[str, Any] = {}
    if department is not None:
        q["department"] = department
    skip = (page - 1) * page_size
    total = db.employees.count_documents(q)
    cursor = (
        db.employees.find(q)
        .sort("emp_code", ASCENDING)
        .skip(skip)
        .limit(page_size)
    )
    items = [serialize_employee(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    ts = ms_to_utc_dt(body.punched_at) if body.punched_at is not None else now_truncated_utc()
    att_date = compute_attendance_date(ts, emp["shift_start"], emp["shift_end"])
    late_mins = compute_late_minutes(ts, emp["shift_start"], att_date)

    doc = {
        "emp_code": body.emp_code,
        "date": att_date,
        "status": body.status.value,
        "punch_in": ts,
        "punch_out": None,
        "work_hours": None,
        "late_minutes": late_mins,
        "overtime_minutes": 0,
        "half_day": False,
        "history": [],
    }
    try:
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError as exc:
        raise HTTPException(status_code=409, detail="Already punched in for this date") from exc

    return serialize_attendance(doc)


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = db.employees.find_one({"emp_code": body.emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    ts = ms_to_utc_dt(body.punched_at) if body.punched_at is not None else now_truncated_utc()

    # Find the employee's most recent record whose punch_in is at or before punched_at
    record = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$ne": None, "$lte": ts}},
        sort=[("punch_in", DESCENDING)],
    )

    if record is None:
        # If no record has punch_in <= ts, check if a record exists with punch_in > ts
        future_record = db.attendance_logs.find_one(
            {"emp_code": body.emp_code, "punch_in": {"$gt": ts}}
        )
        if future_record is not None:
            raise HTTPException(status_code=422, detail="punched_at must be after punch_in")
        raise HTTPException(status_code=404, detail="No punch-in record found")

    # Only treat as an early punch-out (422) for a newer open shift if the previous shift
    # was already closed earlier AND ts belongs to the newer open shift's attendance date
    if record.get("punch_out") is not None and ts > ensure_utc(record["punch_out"]):
        ts_date = compute_attendance_date(ts, emp["shift_start"], emp["shift_end"])
        newer_open = db.attendance_logs.find_one(
            {
                "emp_code": body.emp_code,
                "date": ts_date,
                "punch_in": {"$gt": ts},
                "punch_out": None,
            }
        )
        if newer_open is not None:
            raise HTTPException(status_code=422, detail="punched_at must be after punch_in")

    p_in = ensure_utc(record["punch_in"])
    diff_seconds = (ts - p_in).total_seconds()
    if diff_seconds <= 0 or diff_seconds > 86400:
        raise HTTPException(
            status_code=422,
            detail="punched_at must be strictly after punch_in and within 24 hours",
        )

    if record.get("punch_out") is not None:
        raise HTTPException(status_code=409, detail="Record is already punched out")

    work_hours = compute_work_hours(p_in, ts)
    overtime_mins = compute_overtime(ts, emp["shift_end"], record["date"], emp["shift_start"])
    half_day = work_hours < 4.50

    updated = db.attendance_logs.find_one_and_update(
        {"_id": record["_id"], "punch_out": None},
        {
            "$set": {
                "punch_out": ts,
                "work_hours": work_hours,
                "overtime_minutes": overtime_mins,
                "half_day": half_day,
            }
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(status_code=409, detail="Record is already punched out")

    return serialize_attendance(updated)

@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    q = build_attendance_list_filter(emp_code, date_from, date_to, status)
    skip = (page - 1) * page_size
    total = db.attendance_logs.count_documents(q)
    cursor = (
        db.attendance_logs.find(q)
        .sort([("date", DESCENDING), ("emp_code", ASCENDING)])
        .skip(skip)
        .limit(page_size)
    )
    items = [serialize_attendance(d) for d in cursor]
    return {"items": items, "total": total, "page": page, "page_size": page_size}


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(
    body: RegularizeIn,
    emp_code: str = Path(...),
    date: str = Path(...),
):
    validate_date_param(date, "date")

    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    record = db.attendance_logs.find_one({"emp_code": emp_code, "date": date})
    if not record:
        raise HTTPException(status_code=404, detail="Attendance record not found")

    old_status = record["status"]
    old_punch_in = ensure_utc(record.get("punch_in"))
    old_punch_out = ensure_utc(record.get("punch_out"))
    old_work_hours = record.get("work_hours")
    old_late_minutes = int(record.get("late_minutes") or 0)
    old_overtime_minutes = int(record.get("overtime_minutes") or 0)
    old_half_day = bool(record.get("half_day") or False)

    new_status = body.status.value if body.status is not None else old_status

    if new_status in (Status.ABSENT.value, Status.LEAVE.value):
        if body.punch_in is not None or body.punch_out is not None:
            raise HTTPException(
                status_code=422,
                detail="Cannot supply punch_in or punch_out when status is ABSENT or LEAVE",
            )
        new_punch_in: Optional[datetime] = None
        new_punch_out: Optional[datetime] = None
        new_work_hours: Optional[float] = None
        new_late_minutes = 0
        new_overtime_minutes = 0
        new_half_day = False
    else:
        new_punch_in = ms_to_utc_dt(body.punch_in) if body.punch_in is not None else old_punch_in
        new_punch_out = ms_to_utc_dt(body.punch_out) if body.punch_out is not None else old_punch_out

        if new_punch_in is None:
            raise HTTPException(status_code=422, detail="Presence status requires punch_in")

        calc_date = compute_attendance_date(new_punch_in, emp["shift_start"], emp["shift_end"])
        if calc_date != date:
            raise HTTPException(
                status_code=422,
                detail="punch_in must belong to the record's attendance date",
            )

        if new_punch_out is not None:
            diff_sec = (new_punch_out - new_punch_in).total_seconds()
            if diff_sec <= 0 or diff_sec > 86400:
                raise HTTPException(
                    status_code=422,
                    detail="punch_out must be after punch_in and within 24 hours",
                )

        new_late_minutes = compute_late_minutes(new_punch_in, emp["shift_start"], date)
        if new_punch_out is not None:
            new_work_hours = compute_work_hours(new_punch_in, new_punch_out)
            new_overtime_minutes = compute_overtime(
                new_punch_out, emp["shift_end"], date, emp["shift_start"]
            )
            new_half_day = new_work_hours < 4.50
        else:
            new_work_hours = None
            new_overtime_minutes = 0
            new_half_day = False

    old_fields = {
        "status": old_status,
        "punch_in": old_punch_in,
        "punch_out": old_punch_out,
        "work_hours": old_work_hours,
        "late_minutes": old_late_minutes,
        "overtime_minutes": old_overtime_minutes,
        "half_day": old_half_day,
    }
    new_fields = {
        "status": new_status,
        "punch_in": new_punch_in,
        "punch_out": new_punch_out,
        "work_hours": new_work_hours,
        "late_minutes": new_late_minutes,
        "overtime_minutes": new_overtime_minutes,
        "half_day": new_half_day,
    }

    changes: Dict[str, Dict[str, Any]] = {}
    for k in (
        "status",
        "punch_in",
        "punch_out",
        "work_hours",
        "late_minutes",
        "overtime_minutes",
        "half_day",
    ):
        if old_fields[k] != new_fields[k]:
            changes[k] = {"from": old_fields[k], "to": new_fields[k]}

    if not changes:
        raise HTTPException(status_code=422, detail="Regularization request changes nothing")

    history_entry = {
        "at": now_truncated_utc(),
        "by": body.regularized_by,
        "reason": body.reason,
        "changes": changes,
    }

    concurrency_filter: Dict[str, Any] = {
        "_id": record["_id"],
        "status": record.get("status"),
        "punch_in": record.get("punch_in"),
        "punch_out": record.get("punch_out"),
    }
    if "history" in record:
        concurrency_filter["history"] = record["history"]
    else:
        concurrency_filter["history"] = {"$exists": False}

    updated = db.attendance_logs.find_one_and_update(
        concurrency_filter,
        {
            "$set": new_fields,
            "$push": {"history": history_entry},
        },
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(status_code=409, detail="Concurrent modification conflict")

    return serialize_attendance(updated)


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(
    emp_code: str = Path(...),
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
):
    emp = db.employees.find_one({"emp_code": emp_code})
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")

    month_start, month_end, _, _ = validate_month_param(month)
    working_days = compute_working_days(month_start, month_end, emp["joined_on"])
    pipeline = build_employee_monthly_pipeline(emp_code, month, working_days)
    results = list(db.attendance_logs.aggregate(pipeline))

    if results:
        return results[0]

    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": 0.0,
        "leave_days": 0,
        "late_count": 0,
        "total_late_minutes": 0,
        "total_overtime_minutes": 0,
        "attendance_pct": 0.0 if working_days > 0 else None,
    }


@app.get("/analytics/departments/summary")
def department_summary(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
):
    pipeline = build_department_summary_pipeline(month, department)
    items = list(db.employees.aggregate(pipeline))
    return {"month": month, "items": items}


@app.get("/analytics/leaderboard/late")
def late_leaderboard(
    month: str = Query(..., pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    limit: int = Query(10, ge=1, le=50),
    department: Optional[str] = Query(None),
):
    pipeline = build_late_leaderboard_pipeline(month, limit, department)
    items = list(db.attendance_logs.aggregate(pipeline))
    return {"month": month, "items": items}


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str = Path(...),
    from_date: str = Query(..., alias="from"),
    to_date: str = Query(..., alias="to"),
):
    pipeline = build_department_trend_pipeline(department, from_date, to_date)
    if not db.employees.find_one({"department": department}, {"_id": 1}):
        raise HTTPException(status_code=404, detail="Department not found")

    items = list(db.employees.aggregate(pipeline))
    return {"department": department, "items": items}


@app.get("/admin/explain/{endpoint}")
def explain_endpoint(
    endpoint: ExplainTarget = Path(...),
    emp_code: Optional[str] = Query(None),
    month: Optional[str] = Query(None, pattern=r"^\d{4}-(0[1-9]|1[0-2])$"),
    department: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    status: Optional[Status] = Query(None),
    from_date: Optional[str] = Query(None, alias="from"),
    to_date: Optional[str] = Query(None, alias="to"),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    if endpoint == ExplainTarget.attendance_list:
        q = build_attendance_list_filter(emp_code, date_from, date_to, status)
        collection_name = "attendance_logs"
        cmd = {
            "find": collection_name,
            "filter": q,
            "sort": {"date": -1, "emp_code": 1},
            "skip": (page - 1) * page_size,
            "limit": page_size,
        }
        raw_explain = db.command("explain", cmd, verbosity="executionStats")

    elif endpoint == ExplainTarget.employee_monthly:
        if not emp_code or not month:
            raise HTTPException(
                status_code=422,
                detail="emp_code and month are required for employee_monthly",
            )
        month_start, month_end, _, _ = validate_month_param(month)
        emp = db.employees.find_one({"emp_code": emp_code})
        joined_on = emp["joined_on"] if emp else month_start
        working_days = compute_working_days(month_start, month_end, joined_on)
        pipeline = build_employee_monthly_pipeline(emp_code, month, working_days)
        collection_name = "attendance_logs"
        cmd = {"aggregate": collection_name, "pipeline": pipeline, "cursor": {}}
        raw_explain = db.command("explain", cmd, verbosity="executionStats")

    elif endpoint == ExplainTarget.department_summary:
        if not month:
            raise HTTPException(
                status_code=422,
                detail="month is required for department_summary",
            )
        pipeline = build_department_summary_pipeline(month, department)
        collection_name = "employees"
        cmd = {"aggregate": collection_name, "pipeline": pipeline, "cursor": {}}
        raw_explain = db.command("explain", cmd, verbosity="executionStats")

    elif endpoint == ExplainTarget.late_leaderboard:
        if not month:
            raise HTTPException(
                status_code=422,
                detail="month is required for late_leaderboard",
            )
        pipeline = build_late_leaderboard_pipeline(month, limit, department)
        collection_name = "attendance_logs"
        cmd = {"aggregate": collection_name, "pipeline": pipeline, "cursor": {}}
        raw_explain = db.command("explain", cmd, verbosity="executionStats")

    elif endpoint == ExplainTarget.department_trend:
        if not department or not from_date or not to_date:
            raise HTTPException(
                status_code=422,
                detail="department, from, and to are required for department_trend",
            )
        pipeline = build_department_trend_pipeline(department, from_date, to_date)
        collection_name = "employees"
        cmd = {"aggregate": collection_name, "pipeline": pipeline, "cursor": {}}
        raw_explain = db.command("explain", cmd, verbosity="executionStats")

    return {
        "endpoint": endpoint.value,
        "collection": collection_name,
        "explain": sanitize_bson(raw_explain),
    }