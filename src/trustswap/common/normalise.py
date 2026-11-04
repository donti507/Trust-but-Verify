"""Pure textual normalisation; no schedule or calendar calculations."""
import re
from datetime import date, datetime
from decimal import Decimal
from importlib.resources import files


def normalise_date(value: str | date) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = re.sub(r"(?<=\d)(?:st|nd|rd|th)\b", "", value.strip(), flags=re.I)
    short_year = re.fullmatch(r"(\d{1,2}-[A-Za-z]{3}-)(\d{2})", text)
    if short_year:
        prefix, year = short_year.groups()
        number = int(year)
        text = prefix + str((2000 if number <= 49 else 1900) + number)
    for fmt in ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y/%m/%d", "%d-%b-%Y", "%B %d %Y", "%b %d %Y"):
        try:
            return datetime.strptime(text, fmt).date().isoformat()
        except ValueError:
            pass
    raise ValueError(f"Unsupported or ambiguous date: {value!r}")


def normalise_rate(value: str | Decimal | int | float, *, unit: str | None = None, is_spread: bool = False) -> Decimal:
    """Normalise a quoted rate; explicit no-spread wording requires is_spread=True.

    Missing values and fixed-rate N/A statements never imply a zero rate.
    Bare numeric rates still require an explicit unit.
    """
    text = str(value).strip().lower()
    if is_spread and isinstance(value, str) and " ".join(text.split()) in {"none", "n/a", "not applicable"}:
        return Decimal(0)
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = re.sub(r"^plus\s+", "+", text)
    text = re.sub(r"^minus\s+", "-", text)
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*"
        r"(%|percent|per cent|bps?|basis points?|decimal)?"
        r"(?:\s+(?:per annum|p\.a\.))?", text,
    )
    if not match:
        raise ValueError(f"Invalid rate: {value!r}")
    number, quoted_unit = match.groups()
    aliases = {"%": "percent", "percent": "percent", "per cent": "percent",
               "bp": "bps", "bps": "bps", "basis point": "bps",
               "basis points": "bps", "decimal": "decimal"}
    supplied = aliases.get(unit.strip().lower()) if unit is not None else None
    if unit is not None and supplied is None:
        raise ValueError(f"Unsupported rate unit: {unit!r}")
    quoted = aliases.get(quoted_unit)
    if quoted and supplied and quoted != supplied:
        raise ValueError("Conflicting rate units")
    resolved = quoted or supplied
    if resolved is None:
        raise ValueError("ambiguous rate unit")
    if negative and number.startswith(("+", "-")):
        raise ValueError("Conflicting rate signs")
    amount = Decimal(number) * (-1 if negative else 1)
    return amount / {"percent": Decimal(100), "bps": Decimal(10000), "decimal": Decimal(1)}[resolved]


# Active ISO 4217 currency codes used for monetary amounts.
_CURRENCIES = set("AED AFN ALL AMD ANG AOA ARS AUD AWG AZN BAM BBD BDT BGN BHD BIF BMD BND BOB BOV BRL BSD BTN BWP BYN BZD CAD CDF CHE CHF CHW CLF CLP CNY COP COU CRC CUP CVE CZK DJF DKK DOP DZD EGP ERN ETB EUR FJD FKP GBP GEL GHS GIP GMD GNF GTQ GYD HKD HNL HTG HUF IDR ILS INR IQD IRR ISK JMD JOD JPY KES KGS KHR KMF KPW KRW KWD KYD KZT LAK LBP LKR LRD LSL LYD MAD MDL MGA MKD MMK MNT MOP MRU MUR MVR MWK MXN MXV MYR MZN NAD NGN NIO NOK NPR NZD OMR PAB PEN PGK PHP PKR PLN PYG QAR RON RSD RUB RWF SAR SBD SCR SDG SEK SGD SHP SLE SOS SRD SSP STN SVC SYP SZL THB TJS TMT TND TOP TRY TTD TWD TZS UAH UGX USD USN UYI UYU UYW UZS VED VES VND VUV WST XAF XCD XCG XOF XPF YER ZAR ZMW ZWG".split())


def normalise_notional(value: str, currency: str | None = None) -> tuple[Decimal, str | None]:
    """Return amount and currency; never infer a currency from a bare amount."""
    symbol_codes = {"$": "USD", "US$": "USD", "U.S.$": "USD", "€": "EUR", "£": "GBP"}
    token = r"(?:U\.S\.\$|US\$|\$|€|£|[A-Za-z]{3})"
    match = re.fullmatch(
        rf"\s*({token})?\s*([+-]?(?:\d{{1,3}}(?:,\d{{3}})+|\d+)(?:\.\d+)?)"
        rf"\s*(million|billion|thousand|mm|bn|[kmb])?\s*({token})?\s*", value, re.I,
    )
    if not match:
        raise ValueError(f"Invalid notional: {value!r}")
    prefix, number, scale, suffix = match.groups()
    if scale == "M":
        raise ValueError("ambiguous notional multiplier")
    codes = [symbol_codes.get(c.upper(), c.upper()) for c in (prefix, suffix, currency.strip() if currency else None) if c]
    if len(set(codes)) > 1:
        raise ValueError("Conflicting currencies")
    code = codes[0] if codes else None
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
    text = re.sub(r"[()\[\]]", "", value)
    key = re.sub(r"[\s_.-]", "", text).upper().replace("ACTUAL", "ACT")
    if key == "ACT/365":
        raise ValueError("ambiguous day count")
    return _lookup(key, {
        "ACT/360": "ACT/360", "ACT/365FIXED": "ACT/365.FIXED",
        "ACT/365F": "ACT/365.FIXED", "ACT/ACTISDA": "ACT/ACT.ISDA",
        "ACT/ACTICMA": "ACT/ACT.ICMA", "BONDBASIS": "30/360",
        "30/360BONDBASIS": "30/360", "30/360": "30/360",
        "30E/360": "30E/360", "30E/360ISDA": "30E/360.ISDA",
    })


def normalise_business_day_convention(value: str) -> str:
    text = re.sub(r"\s+Business Day Convention\s*$", "", value.strip(), flags=re.I)
    return _lookup(text, {"FOLLOWING": "FOLLOWING", "MODIFIEDFOLLOWING": "MODFOLLOWING", "MODFOLLOWING": "MODFOLLOWING", "PRECEDING": "PRECEDING", "MODIFIEDPRECEDING": "MODPRECEDING", "MODPRECEDING": "MODPRECEDING", "NONE": "NONE", "UNADJUSTED": "NONE", "NEAREST": "NEAREST"})


def normalise_frequency(value: str) -> str:
    text = re.sub(r"[\s-]", "", value).upper()
    aliases = {"QUARTERLY": "3M", "SEMIANNUAL": "6M", "ANNUAL": "1Y",
               "MONTHLY": "1M", "WEEKLY": "1W", "DAILY": "1D"}
    if text in aliases:
        return aliases[text]
    match = re.fullmatch(r"([1-9]\d*)(D|DAYS?|W|WEEKS?|M|MONTHS?|Y|YEARS?)", text)
    if not match:
        raise ValueError(f"Unsupported frequency: {value!r}")
    count, period = match.groups()
    return count + period[0]


def normalise_business_centers(value: str) -> list[str]:
    aliases = {"LONDON": "GBLO", "NEW YORK": "USNY", "TARGET": "EUTA", "TOKYO": "JPTO"}
    aliases.update({code: code for code in tuple(aliases.values())})
    parts = re.split(r"\s*(?:,|\band\b)\s*", value.strip(), flags=re.I)
    result = set()
    for part in parts:
        key = " ".join(part.upper().split())
        if key not in aliases:
            raise ValueError(f"Unknown business center: {part!r}")
        result.add(aliases[key])
    return sorted(result)


_FLOATING_RATE_CODES = files("trustswap.ref").joinpath("floating_rate_options.txt")


def normalise_floating_rate_option(value: str) -> str:
    """Validate against the local reference list; return an uppercase code."""
    def canonical(text: str) -> str:
        return re.sub(r"\s+", "-", text.strip()).upper()

    code = canonical(value)
    allowed = {canonical(line) for line in _FLOATING_RATE_CODES.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.lstrip().startswith("#")}
    if code not in allowed:
        raise ValueError(f"Unknown floating rate option: {value!r}")
    return code
