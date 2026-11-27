"""A constant-notional, two-stream FpML interest rate swap."""
from xml.etree import ElementTree as ET

from defusedxml import ElementTree as safe_xml
from defusedxml.common import DefusedXmlException

from trustswap.desk.formats import flatten, read_terms

NS = "http://www.fpml.org/FpML-5/confirmation"


def _text(element, *paths):
    return next((text.strip() for path in paths if (text := element.findtext(path)) and text.strip()), None)


def _frequency(stream, path):
    frequency = stream.find(path)
    if frequency is None:
        return None
    multiple = _text(frequency, "periodMultiplier")
    period = _text(frequency, "period")
    if not (multiple and period):
        raise ValueError("Incomplete FpML frequency")
    return multiple + period


def _party(stream, tag, parties):
    element = stream.find(tag)
    href = "" if element is None else element.get("href", "").removeprefix("#")
    if href not in parties:
        raise ValueError("The FpML party reference is missing or unknown")
    return parties[href]


def parse(source):
    try:
        root = safe_xml.fromstring(source, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except (ET.ParseError, DefusedXmlException):
        raise ValueError("Invalid or unsafe FpML document") from None
    for element in root.iter():
        namespace, separator, _ = element.tag.partition("}")
        if separator and not namespace.startswith("{http://www.fpml.org/FpML-"):
            raise ValueError("Unsupported XML namespace")
        element.tag = element.tag.rsplit("}", 1)[-1]
    if root.tag not in {"FpML", "dataDocument", "requestConfirmation",
                        "tradeConfirmed", "trade", "interestRateSwap", "swap"}:
        raise ValueError("Expected an FpML interest rate swap")
    trade_items = [root] if root.tag == "trade" else root.findall(".//trade")
    if len(trade_items) > 1:
        raise ValueError("Upload one FpML trade at a time")
    scope = trade_items[0] if trade_items else root
    swap = scope if scope.tag in {"swap", "interestRateSwap"} else scope.find("swap")
    if swap is None:
        swap = scope.find("interestRateSwap")
    streams = [] if swap is None else swap.findall("swapStream") + swap.findall("interestRateStream")
    if len(streams) != 2:
        raise ValueError("The FpML swap must have exactly two streams")
    if any(swap.findall(f".//{tag}") for tag in (
        "stubCalculationPeriodAmount", "notionalStep", "cashflows", "additionalPayment", "rateCalculation",
    )):
        raise ValueError("The FpML swap contains schedule terms the desk cannot represent")
    if any(node.text and node.text.strip().lower() not in {"false", "0"}
           for exchange in swap.findall(".//principalExchanges") for node in exchange):
        raise ValueError("FpML principal exchanges require manual review")
    parties = {node.get("id"): _text(node, "partyName", "partyId") for node in root.findall(".//party")}
    terms = {"trade_date": _text(scope, "tradeHeader/tradeDate", "tradeDate"),
             "notice_address": _text(scope, "noticeAddress")}
    reference = _text(scope, "tradeHeader/partyTradeIdentifier/tradeId", "tradeId")
    floating = 0
    for stream in streams:
        dates = stream.find("calculationPeriodDates")
        if dates is None:
            raise ValueError("The FpML swap is missing calculation dates")
        for field in ("effective_date", "termination_date"):
            tag = "effectiveDate" if field == "effective_date" else "terminationDate"
            value = _text(dates, f"{tag}/unadjustedDate", f"{tag}/adjustedDate")
            if not value or (field in terms and terms[field] != value):
                raise ValueError("FpML streams must have the same effective and termination dates")
            terms[field] = value
        calc = stream.find("calculationPeriodAmount/calculation")
        if calc is None:
            raise ValueError("The FpML swap is missing a calculation")
        rate = _text(calc, "fixedRateSchedule/initialValue")
        index = _text(calc, "floatingRateCalculation/floatingRateIndex")
        if (rate is None) == (index is None):
            raise ValueError("Expected a fixed or floating rate per FpML stream")
        if rate is not None:
            leg = "fixed_leg"
        else:
            leg = "floating_leg_2" if floating else "floating_leg"
            floating += 1
        if f"{leg}.payer" in terms:
            raise ValueError("Fixed/fixed swaps are outside the desk's supported schema")
        if any(calc.findall(f".//{tag}") for tag in ("step", "capRateSchedule", "floorRateSchedule")):
            raise ValueError("Stepped and capped FpML rates are unsupported")
        notional = "notionalSchedule/notionalStepSchedule"
        conventions = "calculationPeriodDates/calculationPeriodDatesAdjustments"
        centers = [node.text.strip() for node in stream.findall(f"{conventions}/businessCenters/businessCenter")
                   if node.text]
        values = {
            "payer": _party(stream, "payerPartyReference", parties),
            "receiver": _party(stream, "receiverPartyReference", parties),
            "notional_amount": _text(calc, f"{notional}/initialValue"),
            "notional_currency": _text(calc, f"{notional}/currency"),
            "payment_frequency": _frequency(stream, "paymentDates/paymentFrequency"),
            "reset_frequency": _frequency(stream, "resetDates/resetFrequency"),
            "day_count_fraction": _text(calc, "dayCountFraction"),
            "business_day_convention": _text(stream, f"{conventions}/businessDayConvention"),
            "business_centers": ", ".join(centers),
        }
        if rate is not None:
            values["fixed_rate"] = rate
        else:
            values.update(floating_rate_option=index,
                          spread=_text(calc, "floatingRateCalculation/spreadSchedule/initialValue"))
        terms.update({f"{leg}.{name}": value for name, value in values.items()})
    if not terms["trade_date"]:
        raise ValueError("The FpML swap is missing a trade date")
    return read_terms(reference, terms, source)


def _add(parent, path, value):
    """Set a text leaf, sharing its ancestors with the rest of the stream."""
    if value is None:
        return
    *prefix, key = path.split("/")
    for part in prefix:
        node = parent.find(part)
        parent = ET.SubElement(parent, part) if node is None else node
    ET.SubElement(parent, key).text = str(value)


def export(reference, trade):
    """Serialize the typed terms as one FpML confirmation-view trade."""
    fields = flatten(trade)
    root = ET.Element("dataDocument", {"xmlns": NS, "fpmlVersion": "5-12"})
    item = ET.SubElement(root, "trade")
    _add(item, "tradeHeader/partyTradeIdentifier/tradeId", reference)
    _add(item, "tradeHeader/tradeDate", fields.get("trade_date"))
    swap = ET.SubElement(item, "swap")
    party_ids = {}
    for leg in ("fixed_leg", "floating_leg", "floating_leg_2"):
        if getattr(trade, leg) is None:
            continue
        get = lambda name: fields.get(f"{leg}.{name}")
        stream = ET.SubElement(swap, "swapStream")
        for role in ("payer", "receiver"):
            party = get(role)
            if party is None:
                raise ValueError("An FpML export requires both parties on each stream")
            party_id = party_ids.setdefault(party, f"party-{len(party_ids) + 1}")
            ET.SubElement(stream, f"{role}PartyReference", {"href": party_id})
        dates = "calculationPeriodDates"
        _add(stream, f"{dates}/effectiveDate/unadjustedDate", fields.get("effective_date"))
        _add(stream, f"{dates}/terminationDate/unadjustedDate", fields.get("termination_date"))
        adjustments = f"{dates}/calculationPeriodDatesAdjustments"
        _add(stream, f"{adjustments}/businessDayConvention", get("business_day_convention"))
        for center in get("business_centers") or ():
            _add(stream, f"{adjustments}/businessCenters/businessCenter", center)
        for name, path in (("payment_frequency", "paymentDates/paymentFrequency"),
                           ("reset_frequency", "resetDates/resetFrequency")):
            frequency = get(name)
            if frequency:
                _add(stream, f"{path}/periodMultiplier", frequency[:-1])
                _add(stream, f"{path}/period", frequency[-1])
        calc = "calculationPeriodAmount/calculation"
        notional = f"{calc}/notionalSchedule/notionalStepSchedule"
        _add(stream, f"{notional}/initialValue", get("notional_amount"))
        _add(stream, f"{notional}/currency", get("notional_currency"))
        _add(stream, f"{calc}/dayCountFraction", get("day_count_fraction"))
        _add(stream, f"{calc}/fixedRateSchedule/initialValue", get("fixed_rate"))
        floating = f"{calc}/floatingRateCalculation"
        _add(stream, f"{floating}/floatingRateIndex", get("floating_rate_option"))
        _add(stream, f"{floating}/spreadSchedule/initialValue", get("spread"))
    _add(item, "noticeAddress", fields.get("notice_address"))
    for name, party_id in party_ids.items():
        _add(ET.SubElement(root, "party", {"id": party_id}), "partyName", name)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)
