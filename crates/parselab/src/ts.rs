//! Timestamps exactly as rules.md "Timestamps" defines them. Plain code, no cleverness: this is what
//! the fast paths are tested against.

pub const SECOND_MS: i64 = 1_000;
pub const MINUTE_MS: i64 = 60 * SECOND_MS;
pub const HOUR_MS: i64 = 60 * MINUTE_MS;
pub const DAY_MS: i64 = 24 * HOUR_MS;

/// Days since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant's days_from_civil).
pub fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    era * 146_097 + year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year - 719_468
}

/// (year, month, day) of days since 1970-01-01.
pub fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted - era * 146_097;
    let year_of_era =
        (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let month = if month_index < 10 { month_index + 3 } else { month_index - 9 };
    (
        year_of_era + era * 400 + i64::from(month <= 2),
        month,
        day_of_year - (153 * month_index + 2) / 5 + 1,
    )
}

pub fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

pub fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if is_leap_year(year) => 29,
        2 => 28,
        _ => 0,
    }
}

fn digits(s: &[u8]) -> Option<i64> {
    if s.is_empty() || !s.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(s.iter().fold(0i64, |v, &c| v * 10 + (c - b'0') as i64))
}

/// `YYYY-MM-DDTHH:MM:SS` (19 bytes) as ms since the epoch, if every field is in range.
fn date_time(s: &[u8]) -> Option<i64> {
    if s.len() != 19 || s[4] != b'-' || s[7] != b'-' || s[10] != b'T' || s[13] != b':' || s[16] != b':'
    {
        return None;
    }
    let year = digits(&s[0..4])?;
    let month = digits(&s[5..7])?;
    let day = digits(&s[8..10])?;
    let hour = digits(&s[11..13])?;
    let minute = digits(&s[14..16])?;
    let second = digits(&s[17..19])?;
    if !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let days = days_from_civil(year, month, day);
    Some(days * DAY_MS + hour * HOUR_MS + minute * MINUTE_MS + second * SECOND_MS)
}

/// F1 only: `2026-10-03T23:58:10Z`.
pub fn parse_f1(s: &[u8]) -> Option<i64> {
    if s.len() != 20 || s[19] != b'Z' {
        return None;
    }
    date_time(&s[..19])
}

/// Any valid `ts` (F1, F2 or F3) as ms since the epoch. An F3 value too large for i64 is a valid
/// timestamp outside every window; it is returned as i64::MAX.
pub fn parse_ts(s: &[u8]) -> Option<i64> {
    match s.len() {
        20 if s[19] == b'Z' => return date_time(&s[..19]),
        25 if (s[19] == b'+' || s[19] == b'-') && s[22] == b':' => {
            let local = date_time(&s[..19])?;
            let hours = digits(&s[20..22])?;
            let minutes = digits(&s[23..25])?;
            if hours > 23 || minutes > 59 {
                return None;
            }
            let offset = hours * HOUR_MS + minutes * MINUTE_MS;
            return Some(if s[19] == b'+' { local - offset } else { local + offset });
        }
        _ => {}
    }
    // F3: digits only, leading zeros allowed.
    if s.is_empty() || !s.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let significant = &s[s.iter().position(|&c| c != b'0').unwrap_or(s.len())..];
    let mut value: i64 = 0;
    for &c in significant {
        match value.checked_mul(10).and_then(|v| v.checked_add((c - b'0') as i64)) {
            Some(v) => value = v,
            None => return Some(i64::MAX),
        }
    }
    Some(value)
}

/// `price_cents` of a purchase: digits only, value 1 to 10,000,000 (`0250` is 250).
pub fn parse_price(s: &[u8]) -> Option<u32> {
    if s.is_empty() || !s.iter().all(u8::is_ascii_digit) {
        return None;
    }
    let significant = &s[s.iter().position(|&c| c != b'0').unwrap_or(s.len())..];
    if significant.len() > 8 {
        return None;
    }
    let value = significant.iter().fold(0u32, |v, &c| v * 10 + (c - b'0') as u32);
    (1..=10_000_000).contains(&value).then_some(value)
}

/// F1 text of ms since the epoch (whole seconds are printed; the rest is dropped).
pub fn format_f1(ms: i64) -> String {
    let days = ms.div_euclid(DAY_MS);
    let rest = ms.rem_euclid(DAY_MS) / SECOND_MS;
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rest / 3600, rest / 60 % 60, rest % 60)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forms() {
        let f1 = parse_ts(b"2026-10-03T23:58:10Z").unwrap();
        assert_eq!(parse_ts(b"2026-10-03T19:58:10-04:00"), Some(f1));
        assert_eq!(parse_ts(b"1791071890000"), Some(f1));
        assert_eq!(parse_ts(b"0001791071890000"), Some(f1));
        assert_eq!(parse_ts(b"2026-10-04T01:28:10+01:30"), Some(f1));
        assert_eq!(parse_ts(b"2026-10-03T23:58:10-00:00"), Some(f1));
        for bad in [
            &b""[..],
            b"2026-10-03T23:58:10.000Z",
            b"2026-10-03T23:58:10z",
            b"2026-10-03t23:58:10Z",
            b"2026-10-03 23:58:10Z",
            b"2026-10-03T23:58:10",
            b"1791071890000.0",
            b"1.79e12",
            b"-5",
            b"2026-13-03T23:58:10Z",
            b"2026-02-30T23:58:10Z",
            b"2027-02-29T23:58:10Z",
            b"2026-10-03T24:58:10Z",
            b"2026-10-03T23:58:60Z",
            b"2026-10-03T23:58:10+24:00",
            b"2026-10-03T23:58:10+05:60",
            b"2026-10-03T23:58:10+0200",
            b"2026-10-00T23:58:10Z",
            "２026-10-03T23:58:10Z".as_bytes(),
            "١٧٩١٠٧١٨٩٠٠٠٠".as_bytes(),
        ] {
            assert_eq!(parse_ts(bad), None, "{:?}", String::from_utf8_lossy(bad));
        }
        assert!(parse_ts(b"2028-02-29T00:00:00Z").is_some());
        assert_eq!(parse_ts(b"86400000"), Some(86_400_000));
        assert_eq!(parse_ts(b"179102160000000000000"), Some(i64::MAX));
        assert_eq!(parse_ts(b"0"), Some(0));
        assert_eq!(parse_f1(b"1791071890000"), None);
        assert_eq!(format_f1(f1), "2026-10-03T23:58:10Z");
    }

    #[test]
    fn prices() {
        assert_eq!(parse_price(b"0250"), Some(250));
        assert_eq!(parse_price(b"10000000"), Some(10_000_000));
        assert_eq!(parse_price(b"000000000000010000000"), Some(10_000_000));
        assert_eq!(parse_price(b"1"), Some(1));
        for bad in [&b""[..], b"0", b"00", b"10000001", b"250000000", b"19.99", b"-5", b"+500", b"1e3", b"1000\r"]
        {
            assert_eq!(parse_price(bad), None);
        }
    }

    #[test]
    fn civil_roundtrip() {
        for days in -800_000..800_000 {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
            assert!(d >= 1 && d <= days_in_month(y, m));
        }
    }
}
