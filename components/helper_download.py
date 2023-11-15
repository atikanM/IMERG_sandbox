from datetime import datetime, timedelta
from typing import Generator

def date_range(start_date:datetime, stop_date:datetime) -> Generator[datetime, None, None]:
    for n in range(int((stop_date - start_date + timedelta(1)).days)):
        yield start_date + timedelta(n)