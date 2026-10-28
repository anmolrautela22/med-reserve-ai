import unittest
from pathlib import Path
from uuid import uuid4

from med_reserve.database import (
    delete_released_assessments, get_assessments, get_resources, initialize_database, release_allocation, save_assessment,
    update_capacities, validate_assessment,
)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.db_path = Path(__file__).resolve().parent / f".test-db-{uuid4().hex}.sqlite3"
        initialize_database(self.db_path)

    def tearDown(self):
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.db_path) + suffix)
            if path.exists():
                path.unlink()

    def make_assessment(self, person_name="Demo Person", level=2):
        return save_assessment(
            self.db_path, person_name=person_name, age=45, spo2=94, heart_rate=82,
            comorbidities=1, predicted_level=level, predicted_label="Demo category", confidence=0.8,
        )

    def test_initial_resources_and_assessment_persist(self):
        self.assertEqual({row["name"]: row["available"] for row in get_resources(self.db_path)},
                         {"GENERAL": 45, "ICU": 12})
        self.make_assessment()
        initialize_database(self.db_path)
        self.assertEqual({row["name"]: row["available"] for row in get_resources(self.db_path)}["ICU"], 11)
        self.assertEqual(get_assessments(self.db_path)[0]["person_name"], "Demo Person")

    def test_capacity_change_preserves_occupied_slots(self):
        self.make_assessment()
        update_capacities(self.db_path, icu_capacity=20, general_capacity=80)
        resources = {row["name"]: row for row in get_resources(self.db_path)}
        self.assertEqual(resources["ICU"]["available"], 16)
        self.assertEqual(resources["ICU"]["capacity"], 20)
        self.assertEqual(resources["GENERAL"]["available"], 65)
        with self.assertRaises(ValueError):
            update_capacities(self.db_path, icu_capacity=0, general_capacity=80)

    def test_capacity_cannot_be_reduced_below_occupied_slots(self):
        for index in range(4):
            self.make_assessment(f"Demo {index}")
        with self.assertRaises(ValueError):
            update_capacities(self.db_path, icu_capacity=3, general_capacity=60)

    def test_discharge_moves_person_to_released_history_and_returns_bed(self):
        record = self.make_assessment()
        self.assertTrue(release_allocation(self.db_path, record["id"]))
        self.assertFalse(release_allocation(self.db_path, record["id"]))
        saved = get_assessments(self.db_path)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["allocation_status"], "released")
        self.assertEqual(next(row["available"] for row in get_resources(self.db_path) if row["name"] == "ICU"), 12)

    def test_fresh_record_after_release_keeps_both_history_and_active_records(self):
        first = self.make_assessment("Returning Person")
        release_allocation(self.db_path, first["id"])
        second = self.make_assessment("Returning Person")

        records = get_assessments(self.db_path)
        statuses = {row["id"]: row["allocation_status"] for row in records}
        self.assertNotEqual(first["id"], second["id"])
        self.assertEqual(statuses[first["id"]], "released")
        self.assertEqual(statuses[second["id"]], "allocated")
        self.assertEqual(len(records), 2)
    def test_only_released_records_can_be_permanently_removed(self):
        released = self.make_assessment("Released Person")
        release_allocation(self.db_path, released["id"])
        active = self.make_assessment("Active Person")

        removed = delete_released_assessments(self.db_path, [released["id"], active["id"]])

        self.assertEqual(removed, 1)
        remaining = get_assessments(self.db_path)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], active["id"])
        self.assertEqual(remaining[0]["allocation_status"], "allocated")
    def test_capacity_full_is_recorded_without_negative_availability(self):
        for index in range(13):
            record = self.make_assessment(f"Demo {index}")
        self.assertEqual(record["allocation_status"], "capacity_full")
        icu = next(row for row in get_resources(self.db_path) if row["name"] == "ICU")
        self.assertEqual(icu["available"], 0)
        self.assertEqual(len(get_assessments(self.db_path)), 13)

    def test_name_and_input_ranges_are_validated(self):
        self.assertEqual(validate_assessment("  Demo Person  ", 45, 94, 82, 0), "Demo Person")
        with self.assertRaises(ValueError):
            validate_assessment(" ", 45, 94, 82, 0)
        with self.assertRaises(ValueError):
            validate_assessment("Demo", 101, 94, 82, 0)


if __name__ == "__main__":
    unittest.main()



