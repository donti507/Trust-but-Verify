"""Pure textual normalisation; no schedule or calendar calculations."""
import re
from datetime import date, datetime
from decimal import Decimal


def normalise_date(value: str | date) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = value.strip()
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unsupported or ambiguous date: {value!r}")


def normalise_rate(value: str | Decimal | int | float) -> Decimal:
    match = re.fullmatch(r"\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*(%|bps?|basis points?)?\s*", str(value), re.I)
    if not match:
        raise ValueError(f"Invalid rate: {value!r}")
    number, unit = match.groups()
    return Decimal(number) / (Decimal(100) if unit == "%" else Decimal(10000) if unit else Decimal(1))


# Active ISO 4217 currency codes used for monetary amounts.
_CURRENCIES = set("AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BOV BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD USN UYI UYU UYW UZS VED VES VND VUV WST XAF XCD XCG XOF XPF YER ZAR ZMW ZWG".split())


def normalise_notional(value: str, currency: str | None = None) -> tuple[Decimal, str | None]:
    """Return amount and currency; never infer a currency from a bare amount."""
    match = re.fullmatch(r"\s*(?:([A-Za-z]{3})\s+)?([+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)\s*(million|billion|thousand|mm|bn|[kmb])?\s*", value, re.I)
    if not match:
        raise ValueError(f"Invalid notional: {value!r}")
    embedded, number, scale = match.groups()
    embedded = embedded.upper() if embedded else None
    currency = currency.strip().upper() if currency else None
    if embedded and currency and embedded != currency:
        raise ValueError("Conflicting currencies")
    code = embedded or currency
    if code is not None and code not in _CURRENCIES:
        raise ValueError(f"Unknown ISO 4217 currency: {code}")
    multiplier = {None: 1, "k": 1000, "thousand": 1000, "m": 1000000, "mm": 1000000, "million": 1000000, "b": 1000000000, "bn": 1000000000, "billion": 1000000000}[scale.lower() if scale else None]
    return Decimal(number.replace(",", "")) * multiplier, code


def _lookup(value: str, choices: dict[str, str]) -> str:
    key = re.sub(r"[\s_-]", "", value).upper()
    try:
        return choices[key]
    except KeyError:
        raise ValueError(f"Unsupported convention: {value!r}") from None


def normalise_day_count(value: str) -> str:
    return _lookup(value, {"ACTUAL/360": "ACT/360", "ACT/360": "ACT/360", "ACTUAL/365FIXED": "ACT/365.FIXED", "ACT/365.FIXED": "ACT/365.FIXED", "ACT/365F": "ACT/365.FIXED", "ACTUAL/ACTUALISDA": "ACT/ACT.ISDA", "ACT/ACT.ISDA": "ACT/ACT.ISDA", "30/360": "30/360", "30E/360": "30E/360", "30E/360.ISDA": "30E/360.ISDA"})


def normalise_business_day_convention(value: str) -> str:
    return _lookup(value, {"FOLLOWING": "FOLLOWING", "MODIFIEDFOLLOWING": "MODFOLLOWING", "MODFOLLOWING": "MODFOLLOWING", "PRECEDING": "PRECEDING", "MODIFIEDPRECEDING": "MODPRECEDING", "MODPRECEDING": "MODPRECEDING", "NONE": "NONE", "UNADJUSTED": "NONE", "NEAREST": "NEAREST"})
