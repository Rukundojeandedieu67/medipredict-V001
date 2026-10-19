"""Create the Phase 1 PostGIS schema."""

from alembic import op


revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


TABLES = (
    """
    CREATE TABLE districts (
        district_id text PRIMARY KEY,
        name text NOT NULL UNIQUE,
        province text NOT NULL,
        geom geometry(Point, 4326),
        attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
        data_source data_source_enum NOT NULL,
        CHECK (district_id <> ''),
        CHECK (name <> '')
    )
    """,
    """
    CREATE TABLE facilities (
        facility_id text PRIMARY KEY,
        district_id text NOT NULL REFERENCES districts(district_id),
        name text,
        tier text,
        amenity text,
        healthcare text,
        geom geometry(Point, 4326),
        attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
        data_source data_source_enum NOT NULL
    )
    """,
    """
    CREATE TABLE nodes (
        node_id text PRIMARY KEY,
        district_id text NOT NULL REFERENCES districts(district_id),
        facility_id text REFERENCES facilities(facility_id),
        name text NOT NULL,
        node_type text NOT NULL CHECK (node_type IN ('district_blood_bank', 'kigali_referral_hospital')),
        enabled boolean NOT NULL DEFAULT true,
        attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
        data_source data_source_enum NOT NULL
    )
    """,
    """
    CREATE TABLE district_distance (
        from_district text NOT NULL REFERENCES districts(district_id),
        to_district text NOT NULL REFERENCES districts(district_id),
        straight_km double precision NOT NULL CHECK (straight_km >= 0),
        est_road_km double precision CHECK (est_road_km IS NULL OR est_road_km >= 0),
        est_transit_hours double precision CHECK (est_transit_hours IS NULL OR est_transit_hours >= 0),
        data_source data_source_enum NOT NULL,
        PRIMARY KEY (from_district, to_district)
    )
    """,
    """
    CREATE TABLE blood_groups (
        code text PRIMARY KEY,
        rbc_2024_share double precision NOT NULL CHECK (rbc_2024_share >= 0 AND rbc_2024_share <= 1),
        data_source data_source_enum NOT NULL
    )
    """,
    """
    CREATE TABLE ref_hospitalisation (
        id bigserial PRIMARY KEY,
        district_id text NOT NULL REFERENCES districts(district_id),
        facility_type text NOT NULL,
        admissions integer NOT NULL CHECK (admissions >= 0),
        avg_los_days double precision CHECK (avg_los_days IS NULL OR avg_los_days >= 0),
        source_period text,
        attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
        data_source data_source_enum NOT NULL
    )
    """,
    """
    CREATE TABLE ref_blood_usage (
        id bigserial PRIMARY KEY,
        district_id text NOT NULL REFERENCES districts(district_id),
        blood_group text NOT NULL REFERENCES blood_groups(code),
        units integer NOT NULL CHECK (units >= 0),
        source_period text,
        attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
        data_source data_source_enum NOT NULL
    )
    """,
    """
    CREATE TABLE ref_rbc_2024 (
        id bigserial PRIMARY KEY,
        record_type text NOT NULL,
        record_key text NOT NULL,
        year integer NOT NULL,
        blood_group text REFERENCES blood_groups(code),
        region text,
        rcbt text,
        indicator text,
        indication text,
        units bigint CHECK (units IS NULL OR units >= 0),
        components jsonb NOT NULL DEFAULT '{}'::jsonb,
        source text NOT NULL,
        attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
        data_source data_source_enum NOT NULL,
        UNIQUE (record_type, record_key)
    )
    """,
    """
    CREATE TABLE ref_blood_group_mix_audit (
        blood_group text PRIMARY KEY REFERENCES blood_groups(code),
        attributes jsonb NOT NULL,
        data_source data_source_enum NOT NULL CHECK (data_source = 'derived')
    )
    """,
    """
    CREATE TABLE stock_readings (
        id bigserial PRIMARY KEY,
        node_id text NOT NULL REFERENCES nodes(node_id),
        district_id text NOT NULL REFERENCES districts(district_id),
        blood_group text NOT NULL REFERENCES blood_groups(code),
        observed_at timestamptz NOT NULL,
        stock_open_units integer NOT NULL CHECK (stock_open_units >= 0),
        stock_units integer NOT NULL CHECK (stock_units >= 0),
        demand_units integer NOT NULL CHECK (demand_units >= 0),
        supply_received integer NOT NULL CHECK (supply_received >= 0),
        expired_units integer NOT NULL CHECK (expired_units >= 0),
        unmet_demand_units integer NOT NULL CHECK (unmet_demand_units >= 0),
        lead_time_hours double precision NOT NULL CHECK (lead_time_hours >= 0),
        delivery_mode text NOT NULL CHECK (delivery_mode IN ('none', 'opening', 'road', 'drone')),
        data_source data_source_enum NOT NULL,
        UNIQUE (node_id, blood_group, observed_at)
    )
    """,
    """
    CREATE TABLE demand_events (
        event_id text PRIMARY KEY,
        occurred_at timestamptz NOT NULL,
        node_id text NOT NULL REFERENCES nodes(node_id),
        district_id text NOT NULL REFERENCES districts(district_id),
        blood_group text NOT NULL REFERENCES blood_groups(code),
        event_type text NOT NULL,
        units integer NOT NULL CHECK (units > 0),
        data_source data_source_enum NOT NULL
    )
    """,
    """
    CREATE TABLE transfers (
        id bigserial PRIMARY KEY,
        source_node_id text REFERENCES nodes(node_id),
        destination_node_id text REFERENCES nodes(node_id),
        source_district text NOT NULL REFERENCES districts(district_id),
        destination_district text NOT NULL REFERENCES districts(district_id),
        blood_group text NOT NULL REFERENCES blood_groups(code),
        units integer NOT NULL CHECK (units > 0),
        estimated_transit_hours double precision NOT NULL CHECK (estimated_transit_hours >= 0),
        status text NOT NULL CHECK (status IN ('proposed', 'approved', 'rejected', 'completed', 'cancelled')),
        created_at timestamptz NOT NULL DEFAULT now(),
        data_source data_source_enum NOT NULL,
        CHECK (source_district <> destination_district)
    )
    """,
    """
    CREATE TABLE predictions (
        id bigserial PRIMARY KEY,
        node_id text NOT NULL REFERENCES nodes(node_id),
        district_id text NOT NULL REFERENCES districts(district_id),
        blood_group text NOT NULL REFERENCES blood_groups(code),
        predicted_at timestamptz NOT NULL,
        horizon_hours integer NOT NULL CHECK (horizon_hours IN (24, 48)),
        risk_probability double precision NOT NULL CHECK (risk_probability >= 0 AND risk_probability <= 1),
        model_version text NOT NULL,
        data_source data_source_enum NOT NULL
    )
    """,
    """
    CREATE TABLE recommendations (
        id bigserial PRIMARY KEY,
        transfer_id bigint REFERENCES transfers(id),
        source_node_id text REFERENCES nodes(node_id),
        destination_node_id text REFERENCES nodes(node_id),
        source_district text NOT NULL REFERENCES districts(district_id),
        destination_district text NOT NULL REFERENCES districts(district_id),
        blood_group text NOT NULL REFERENCES blood_groups(code),
        units integer NOT NULL CHECK (units > 0),
        estimated_transit_hours double precision NOT NULL CHECK (estimated_transit_hours >= 0),
        expected_benefit_units double precision NOT NULL CHECK (expected_benefit_units >= 0),
        reason text NOT NULL,
        status text NOT NULL CHECK (status IN ('open', 'approved', 'rejected', 'expired')),
        created_at timestamptz NOT NULL DEFAULT now(),
        data_source data_source_enum NOT NULL,
        CHECK (source_district <> destination_district)
    )
    """,
    """
    CREATE TABLE decisions_audit (
        id bigserial PRIMARY KEY,
        recommendation_id bigint NOT NULL REFERENCES recommendations(id),
        decision text NOT NULL CHECK (decision IN ('approve', 'reject')),
        actor text NOT NULL CHECK (actor <> ''),
        note text NOT NULL DEFAULT '',
        decided_at timestamptz NOT NULL DEFAULT now(),
        data_source data_source_enum NOT NULL CHECK (data_source = 'entered')
    )
    """,
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")
    op.execute("CREATE TYPE data_source_enum AS ENUM ('real', 'derived', 'simulated', 'entered')")
    for statement in TABLES:
        op.execute(statement)
    op.execute("CREATE INDEX facilities_geom_gist ON facilities USING GIST (geom)")
    op.execute("CREATE INDEX districts_geom_gist ON districts USING GIST (geom)")
    op.execute("CREATE INDEX stock_readings_observed_at_idx ON stock_readings (observed_at)")
    op.execute("CREATE INDEX predictions_risk_idx ON predictions (risk_probability DESC)")
    op.execute("CREATE INDEX recommendations_status_idx ON recommendations (status, created_at)")


def downgrade() -> None:
    for table in (
        "decisions_audit",
        "recommendations",
        "predictions",
        "transfers",
        "demand_events",
        "stock_readings",
        "ref_blood_group_mix_audit",
        "ref_rbc_2024",
        "ref_blood_usage",
        "ref_hospitalisation",
        "blood_groups",
        "district_distance",
        "nodes",
        "facilities",
        "districts",
    ):
        op.execute(f"DROP TABLE {table}")
    op.execute("DROP TYPE data_source_enum")
