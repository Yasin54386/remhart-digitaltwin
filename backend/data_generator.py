from app.db import SessionLocal
from app.models.db_model import DateTimeTable, VoltageTable, CurrentTable, FrequencyTable, ActivePowerTable, ReactivePowerTable
from app.grid_generator import grid_generator  # your data generator

db = SessionLocal()

# Generate 100 time-series data points
data_points = grid_generator.generate_time_series(100)

for dp in data_points:
    dt = DateTimeTable(timestamp=dp['timestamp'])
    db.add(dt)
    db.flush()  # to get dt.id

    db.add(VoltageTable(timestamp_id=dt.id, **dp['voltage']))
    db.add(CurrentTable(timestamp_id=dt.id, **dp['current']))
    db.add(FrequencyTable(timestamp_id=dt.id, **dp['frequency']))
    db.add(ActivePowerTable(timestamp_id=dt.id, **dp['active_power']))
    db.add(ReactivePowerTable(timestamp_id=dt.id, **dp['reactive_power']))

db.commit()
print("Grid data generated successfully!")
