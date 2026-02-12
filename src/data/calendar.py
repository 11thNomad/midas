"""NSE market calendar utilities for trading-day-aware computations."""

from __future__ import annotations

from datetime import date, timedelta


# NSE holidays (maintained manually; update yearly from NSE circular)
HOLIDAYS: dict[int, list[date]] = {
    2023: [
        date(2023, 1, 26),
        date(2023, 3, 7),
        date(2023, 3, 30),
        date(2023, 4, 4),
        date(2023, 4, 7),
        date(2023, 4, 14),
        date(2023, 4, 22),
        date(2023, 5, 1),
        date(2023, 6, 29),
        date(2023, 8, 15),
        date(2023, 9, 19),
        date(2023, 10, 2),
        date(2023, 10, 24),
        date(2023, 11, 14),
        date(2023, 11, 27),
        date(2023, 12, 25),
    ],
    2024: [
        date(2024, 1, 26),
        date(2024, 3, 8),
        date(2024, 3, 25),
        date(2024, 3, 29),
        date(2024, 4, 11),
        date(2024, 4, 17),
        date(2024, 5, 1),
        date(2024, 5, 23),
        date(2024, 6, 17),
        date(2024, 7, 17),
        date(2024, 8, 15),
        date(2024, 10, 2),
        date(2024, 11, 1),
        date(2024, 11, 15),
        date(2024, 12, 25),
    ],
    2025: [
        date(2025, 2, 26),
        date(2025, 3, 14),
        date(2025, 3, 31),
        date(2025, 4, 10),
        date(2025, 4, 14),
        date(2025, 4, 18),
        date(2025, 5, 1),
        date(2025, 8, 15),
        date(2025, 8, 27),
        date(2025, 10, 2),
        date(2025, 10, 21),
        date(2025, 10, 22),
        date(2025, 11, 5),
        date(2025, 12, 25),
    ],
    2026: [
        date(2026, 1, 26),
        date(2026, 2, 17),
        date(2026, 3, 4),
        date(2026, 3, 20),
        date(2026, 4, 3),
        date(2026, 4, 14),
        date(2026, 5, 1),
        date(2026, 8, 15),
        date(2026, 10, 2),
        date(2026, 11, 9),
        date(2026, 12, 25),
    ],
}


class NSECalendar:
    """Minimal trading-day calendar for NSE cash/F&O quality checks."""

    def __init__(self):
        self._holiday_set: set[date] = set()
        for values in HOLIDAYS.values():
            self._holiday_set.update(values)

    @staticmethod
    def is_weekend(day: date) -> bool:
        return day.weekday() >= 5

    def is_holiday(self, day: date) -> bool:
        return day in self._holiday_set

    def is_trading_day(self, day: date) -> bool:
        return not self.is_weekend(day) and not self.is_holiday(day)

    def trading_days_between(self, start: date, end: date) -> list[date]:
        current = start
        out: list[date] = []
        while current <= end:
            if self.is_trading_day(current):
                out.append(current)
            current += timedelta(days=1)
        return out


nse_calendar = NSECalendar()
