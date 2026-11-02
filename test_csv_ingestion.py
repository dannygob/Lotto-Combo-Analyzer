from src.csv_learning import import_and_retrain_csv, preview_csv_schema
from src.domain import Ruleset, NumberPool, PoolKind, Lottery
from src.registry import SQLiteRegistry
from datetime import date

# Create registry and register ruleset
registry = SQLiteRegistry(r'data\lotto_registry.db')
registry.initialize()

# Use a new lottery ID to avoid conflicts
lottery_id = 'lotto-new'

try:
    registry.register_lottery(Lottery(lottery_id=lottery_id, name='New Test Lottery'))
except:
    pass

ruleset = Ruleset(
    lottery_id=lottery_id,
    version='v1',
    effective_from=date(2020, 1, 1),
    effective_to=date(2025, 12, 31),
    pools=(
        NumberPool(pool_id='principal', kind=PoolKind.PRIMARY, minimum=1, maximum=52, numbers_drawn=5),
        NumberPool(pool_id='suplementario', kind=PoolKind.PRIMARY, minimum=1, maximum=10, numbers_drawn=2),
    )
)
registry.register_ruleset(ruleset)
print('Ruleset registered for testing')

# Create a test CSV with the format:
# - Date column
# - 5 number columns for the main pool
# - 2 number columns for the supplementary pool
# - Event ID column
csv_content = """sorteo,fecha,numero_1,numero_2,numero_3,numero_4,numero_5,complement_1,complement_2
001,2024-01-15,5,12,23,34,45,1,15
002,2024-01-22,8,15,30,38,42,3,19
003,2024-01-29,3,9,27,14,26,7,21
"""

print('\n--- Testing CSV preview (schema suggestion) ---')
# Test preview - this will suggest the column mapping
preview = preview_csv_schema(
    raw_csv=csv_content.encode('utf-8'),
    filename='test_lottery.csv',
    lottery_id=lottery_id,
    ruleset=ruleset
)
print('Schema preview results:')
print('  - Rows:', preview['rows'])
print('  - Number column candidates:', preview['number_column_candidates'])
print('  - Suggested pool columns:', preview['suggested_pool_columns'])
print('  - Mapping unambiguous:', preview['mapping_unambiguous'])
print('  - Confirmation required:', preview['confirmation_required'])
print('  - Other columns:', preview['other_columns'])
print('  - Notice:', preview['notice'])

print('\n--- Testing full import with confirmed mapping ---')
# The preview shows confirmation_required is True, meaning we need to confirm the mapping
# Let's try the import with explicit pool_columns based on preview
try:
    result = import_and_retrain_csv(
        registry=registry,
        raw_csv=csv_content.encode('utf-8'),
        filename='test_lottery.csv',
        lottery_id=lottery_id,
        ruleset_version='v1',
        data_directory='data/datasets',
        model_directory='data/models',
        # Based on preview, we need to provide explicit pool_columns
        # The preview should tell us the suggested columns
    )
    print('CSV import successful!')
    print('  - Status:', result['status'])
    print('  - Model status:', result['model_status'])
    print('  - Row count:', result['row_count'])
    print('  - Number columns:', result['number_columns'])
    print('  - Blank rows skipped:', result['blank_rows_skipped'])
    print('  - Rows without draw numbers skipped:', result['rows_without_draw_numbers_skipped'])
    print('  - Metrics:', result['metrics'])
except Exception as e:
    print('CSV import failed:', e)