from datetime import date
from src.domain import Ruleset, NumberPool, PoolKind, Lottery
from src.registry import SQLiteRegistry

# Create registry instance
registry = SQLiteRegistry(r'data\lotto_registry.db')
registry.initialize()

# Register lottery (skip if it already exists)
try:
    registry.register_lottery(Lottery(lottery_id='test-lottery', name='Test Lottery'))
    print('✅ Lottery registered')
except Exception as e:
    print(f'Lottery may already exist: {e}')

# Register ruleset with BOTH pools (5-number + 2-number)
ruleset = Ruleset(
    lottery_id='test-lottery',
    version='v1',
    effective_from=date(2020, 1, 1),
    effective_to=date(2025, 12, 31),
    pools=(
        # Primary pool: 5 numbers from range 1–52
        NumberPool(
            pool_id='principal',
            kind=PoolKind.PRIMARY,
            minimum=1,
            maximum=52,
            numbers_drawn=5
        ),
        # Supplementary pool: 2 numbers from range 1–10
        NumberPool(
            pool_id='suplementario',
            kind=PoolKind.PRIMARY,
            minimum=1,
            maximum=10,
            numbers_drawn=2
        )
    )
)

# Register ruleset in the registry
registry.register_ruleset(ruleset)
print('✅ Ruleset registered successfully!')
print(f'   Pools: {len(ruleset.pools)} pool(s)')

# Print pool details
for pool in ruleset.pools:
    print(f'   - Pool {pool.pool_id}: {pool.numbers_drawn} numbers, range {pool.minimum}-{pool.maximum}')
