//! Timestamps: the three forms of rules.md to integer milliseconds since the epoch, UTC.
//!
//! F1 `YYYY-MM-DDTHH:MM:SSZ`, F2 `YYYY-MM-DDTHH:MM:SS+HH:MM` (or `-`), F3 ASCII digits only.
//! Everything else is `None` (bad). An F3 value too large for i64 is also `None`: the rules
//! call it valid but outside every window, and a bad and an out-of-window `ts` both drop the row.

pub const SECOND: i64 = 1_000;
pub const MINUTE: i64 = 60 * SECOND;
pub const HOUR: i64 = 60 * MINUTE;
pub const DAY: i64 = 24 * HOUR;

/// Days from 1970-01-01 to the given proleptic Gregorian date (Howard Hinnant's algorithm).
pub fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let year = if month <= 2 { year - 1 } else { year };
    let era = year.div_euclid(400);
    let year_of_era = year - era * 400;
    let day_of_year = (153 * ((month + 9) % 12) + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    era * 146_097 + day_of_era - 719_468
}

fn is_leap(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ => {
            if is_leap(year) {
                29
            } else {
                28
            }
        }
    }
}

#[inline]
fn two(s: &[u8], i: usize) -> Option<i64> {
    let a = s[i].wrapping_sub(b'0');
    let b = s[i + 1].wrapping_sub(b'0');
    if a < 10 && b < 10 { Some(i64::from(a) * 10 + i64::from(b)) } else { None }
}

#[inline]
fn epoch_digits(s: &[u8]) -> Option<i64> {
    let mut value: i64 = 0;
    for &c in s {
        let d = c.wrapping_sub(b'0');
        if d >= 10 {
            return None;
        }
        value = value.checked_mul(10)?.checked_add(i64::from(d))?;
    }
    Some(value)
}

/// Parses a `ts` value; `None` means bad.
#[inline]
pub fn parse(s: &[u8]) -> Option<i64> {
    let n = s.len();
    if n == 0 {
        return None;
    }
    if n != 20 && n != 25 {
        return epoch_digits(s);
    }
    if s[4] != b'-' {
        // 20 or 25 characters with no date shape: only F3 digits can still be valid.
        return epoch_digits(s);
    }
    if s[7] != b'-' || s[10] != b'T' || s[13] != b':' || s[16] != b':' {
        return None;
    }
    let year = two(s, 0)? * 100 + two(s, 2)?;
    let month = two(s, 5)?;
    let day = two(s, 8)?;
    let hour = two(s, 11)?;
    let minute = two(s, 14)?;
    let second = two(s, 17)?;
    if !(1..=12).contains(&month) || day < 1 || day > days_in_month(year, month) {
        return None;
    }
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let local = (((days_from_civil(year, month, day) * 24 + hour) * 60 + minute) * 60 + second) * SECOND;
    if n == 20 {
        return if s[19] == b'Z' { Some(local) } else { None };
    }
    let sign = match s[19] {
        b'+' => 1,
        b'-' => -1,
        _ => return None,
    };
    if s[22] != b':' {
        return None;
    }
    let offset_hours = two(s, 20)?;
    let offset_minutes = two(s, 23)?;
    if offset_hours > 23 || offset_minutes > 59 {
        return None;
    }
    Some(local - sign * (offset_hours * 60 + offset_minutes) * MINUTE)
}

/// Parses a purchase price: ASCII digits only, value 1 to 10,000,000. `None` means bad.
#[inline]
pub fn price(s: &[u8]) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    let mut value: u64 = 0;
    for &c in s {
        let d = c.wrapping_sub(b'0');
        if d >= 10 {
            return None;
        }
        value = value * 10 + u64::from(d);
        if value > 10_000_000 {
            // Later digits only grow it, but every byte must still be a digit for a valid
            // price; an over-range value is bad either way, so stop here.
            return None;
        }
    }
    if value >= 1 { Some(value as u32) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forms() {
        let f1 = parse(b"2026-10-03T23:58:10Z").unwrap();
        assert_eq!(f1, 1_791_071_890_000);
        assert_eq!(parse(b"2026-10-03T19:58:10-04:00"), Some(f1));
        assert_eq!(parse(b"1791071890000"), Some(f1));
        assert_eq!(parse(b"0001791071890000"), Some(f1));
        assert_eq!(parse(b"00000000000000000000000001791071890000"), Some(f1));
        assert_eq!(parse(b"2026-10-04T01:30:00+02:00"), parse(b"2026-10-03T23:30:00Z"));
        assert_eq!(parse(b"2026-10-03T10:00:00-00:00"), parse(b"2026-10-03T10:00:00Z"));
        assert_eq!(parse(b"2026-10-03T10:00:00+23:59"), Some(parse(b"2026-10-03T10:00:00Z").unwrap() - 1439 * MINUTE));
        assert_eq!(parse(b"86400000"), Some(DAY));
        assert_eq!(parse(b"2028-02-29T00:00:00Z"), Some(days_from_civil(2028, 2, 29) * DAY));
    }

    #[test]
    fn bad_forms() {
        for bad in [
            &b""[..], b"2026-10-03T23:58:10.000Z", b"2026-10-03T23:58:10z", b"2026-10-03t23:58:10Z",
            b"2026-10-03 23:58:10Z", b"2026-10-03T23:58:10", b"1791071890000.0", b"1.79e12", b"-5",
            b"2026-13-03T23:58:10Z", b"2026-02-30T23:58:10Z", b"2027-02-29T23:58:10Z",
            b"2026-10-03T24:00:00Z", b"2026-10-03T23:58:60Z", b"2026-10-03T23:58:10+24:00",
            b"2026-10-03T23:58:10+05:60", b"2026-10-03T23:58:10+0200", b"179102160000000000000",
            b"2026-10-03T23:58:10Z\r", "\u{ff12}026-10-03T10:00:00Z".as_bytes(),
            "\u{662}\u{665}\u{660}".as_bytes(), b"+1791071890000", b" 1791071890000",
            b"2026-10-03T23:58:10+02:00Z", b"2026-1-03T23:58:10Z", b"12345678901234567890",
        ] {
            assert_eq!(parse(bad), None, "{:?}", String::from_utf8_lossy(bad));
        }
        // 20 digits that fit in i64 are F3, not a broken F1.
        assert_eq!(parse(b"00000001791071890000"), Some(1_791_071_890_000));
        assert_eq!(parse(b"0000000000001791071890000"), Some(1_791_071_890_000));
    }

    #[test]
    fn prices() {
        assert_eq!(price(b"0250"), Some(250));
        assert_eq!(price(b"10000000"), Some(10_000_000));
        assert_eq!(price(b"0000000000000000000010000000"), Some(10_000_000));
        assert_eq!(price(b"10000001"), None);
        assert_eq!(price(b"0"), None);
        assert_eq!(price(b""), None);
        assert_eq!(price(b"19.99"), None);
        assert_eq!(price(b"-5"), None);
        assert_eq!(price(b"1000\r"), None);
        assert_eq!(price("\u{ff12}\u{ff15}\u{ff10}".as_bytes()), None);
    }
}
