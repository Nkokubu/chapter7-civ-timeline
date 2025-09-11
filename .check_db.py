from sqlalchemy import inspect
from app.db.session import engine
insp = inspect(engine)
print("tables:", insp.get_table_names())
print("sources cols:", [c['name'] for c in insp.get_columns('sources')])
print("civ_source_link cols:", [c['name'] for c in insp.get_columns('civ_source_link')])
print("event_source_link cols:", [c['name'] for c in insp.get_columns('event_source_link')])
