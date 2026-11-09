
import unittest
from unittest.mock import patch

import numpy as np

from agentdriver.planning.motion_planning import planning_single_inference


class TestQwenTrajectoryParser(unittest.TestCase):

    def check_trajectory(self, response, expected):
        """Test the actual AgentDriver trajectory extraction."""

        with patch(
            "agentdriver.planning.motion_planning.generate_messages",
            return_value=("test_token", "test prompt", None)
        ), patch(
            "agentdriver.planning.motion_planning.run_one_round_conversation",
            return_value=([], {"content": response})
        ):
            traj, _ = planning_single_inference(
                planner_model_id="qwen3:14b",
                data_sample={},
                self_reflection=False,
                verbose=False,
            )

        np.testing.assert_allclose(
            traj,
            np.array(expected, dtype=float),
            atol=1e-6,
        )
        self.assertEqual(traj.shape, (6, 2))

    def test_python_list(self):
        """Format A: Original AgentDriver format."""

        expected = [
            (0.028, 1.13),
            (0.040, 2.20),
            (0.038, 3.21),
            (0.020, 4.16),
            (-0.013, 5.05),
            (-0.060, 5.88),
        ]

        response = f"Planned Trajectory:\n{expected}"
        self.check_trajectory(response, expected)

    def test_timestamped_coordinates(self):
        """Format B: Scenario 2 from the 20-scenario run."""

        response = """
- **0.5 seconds**:
  $ (x, y) = (0.028, 1.13) $
- **1.0 seconds**:
  $ (x, y) = (0.040, 2.20) $
- **1.5 seconds**:
  $ (x, y) = (0.038, 3.21) $
- **2.0 seconds**:
  $ (x, y) = (0.020, 4.16) $
- **2.5 seconds**:
  $ (x, y) = (-0.013, 5.05) $
- **3.0 seconds**:
  $ (x, y) = (-0.060, 5.88) $
"""

        expected = [
            (0.028, 1.13),
            (0.040, 2.20),
            (0.038, 3.21),
            (0.020, 4.16),
            (-0.013, 5.05),
            (-0.060, 5.88),
        ]

        self.check_trajectory(response, expected)

    def test_numbered_waypoints(self):
        """Format C: Scenario 4 from the 20-scenario run."""

        response = """
The following waypoints are at 0.5s, 1.0s,
1.5s, 2.0s, 2.5s, and 3.0s.

**Waypoints:**
1. **(-0.02, 2.16)**
2. **(-0.03, 4.30)**
3. **(-0.03, 6.43)**
4. **(-0.02, 8.54)**
5. **(0.00, 10.64)**
6. **(0.03, 12.72)**
"""

        expected = [
            (-0.02, 2.16),
            (-0.03, 4.30),
            (-0.03, 6.43),
            (-0.02, 8.54),
            (0.00, 10.64),
            (0.03, 12.72),
        ]

        self.check_trajectory(response, expected)

    def test_numbered_waypoints_with_interval(self):
        """Format C: Scenario 6 from the 20-scenario run."""

        response = """
**Planned Trajectory Waypoints:**

1. **(-0.02, 1.51)**
2. **(-0.07, 3.02)**
3. **(-0.17, 4.53)**
4. **(-0.30, 6.04)**
5. **(-0.48, 7.55)**
6. **(-0.69, 9.06)**

The trajectory uses six future waypoints
at 0.5s intervals, starting at 0.5s.
"""

        expected = [
            (-0.02, 1.51),
            (-0.07, 3.02),
            (-0.17, 4.53),
            (-0.30, 6.04),
            (-0.48, 7.55),
            (-0.69, 9.06),
        ]

        self.check_trajectory(response, expected)

    def test_separate_xy_values(self):
        """Format D: Scenario 20 from the 20-scenario run."""

        response = r"""
- **At 0.5 seconds**:
  $ x \approx 0.004 \, \text{m}, y \approx 1.45 \, \text{m} $
- **At 1.0 seconds**:
  $ x \approx 0.005 \, \text{m}, y \approx 2.70 \, \text{m} $
- **At 1.5 seconds**:
  $ x \approx 0.006 \, \text{m}, y \approx 3.74 \, \text{m} $
- **At 2.0 seconds**:
  $ x \approx 0.007 \, \text{m}, y \approx 4.60 \, \text{m} $
- **At 2.5 seconds**:
  $ x \approx 0.008 \, \text{m}, y \approx 5.26 \, \text{m} $
- **At 3.0 seconds**:
  $ x \approx 0.009 \, \text{m}, y \approx 5.72 \, \text{m} $
"""

        expected = [
            (0.004, 1.45),
            (0.005, 2.70),
            (0.006, 3.74),
            (0.007, 4.60),
            (0.008, 5.26),
            (0.009, 5.72),
        ]

        self.check_trajectory(response, expected)

    def test_incorrect_timestamps(self):
        """Reject a trajectory that starts at t=0."""

        response = """
1. **At 0s:**
   **Waypoint:** $ (0, 0) $
2. **At 0.5s:**
   **Waypoint:** $ (0, 1.446) $
3. **At 1.0s:**
   **Waypoint:** $ (0, 2.695) $
4. **At 1.5s:**
   **Waypoint:** $ (0, 3.744) $
5. **At 2.0s:**
   **Waypoint:** $ (0, 4.600) $
6. **At 2.5s:**
   **Waypoint:** $ (0, 5.264) $
"""

        with self.assertRaises(ValueError):
            self.check_trajectory(response, [])

    def test_missing_waypoint(self):
        """Reject trajectories containing fewer than six points."""

        response = """
[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
"""

        with self.assertRaises(ValueError):
            self.check_trajectory(response, [])

    def test_non_finite_coordinates(self):
        """Reject infinite coordinates."""

        response = """
[(0, 1), (0, 2), (0, 3),
 (0, 4), (0, 5), (float('inf'), 6)]
"""

        with self.assertRaises(ValueError):
            self.check_trajectory(response, [])



    def test_ambiguous_numbered_waypoints(self):
        """Reject numbered coordinates without timing information."""
        response = """
1. **(0, 1)**
2. **(0, 2)**
3. **(0, 3)**
4. **(0, 4)**
5. **(0, 5)**
6. **(0, 6)**
"""
        with self.assertRaises(ValueError):
            self.check_trajectory(response, [])

    def test_incorrect_explicit_timestamps(self):
        """Reject timestamped coordinates starting at t=0."""
        response = """
- **0.0 seconds**:
  $ (x, y) = (0, 0) $
- **0.5 seconds**:
  $ (x, y) = (0, 1) $
- **1.0 seconds**:
  $ (x, y) = (0, 2) $
- **1.5 seconds**:
  $ (x, y) = (0, 3) $
- **2.0 seconds**:
  $ (x, y) = (0, 4) $
- **2.5 seconds**:
  $ (x, y) = (0, 5) $
"""
        with self.assertRaises(ValueError):
            self.check_trajectory(response, [])

    def test_incomplete_numbered_waypoints(self):
        """Reject a numbered trajectory missing waypoint 6."""
        response = """
Waypoints at 0.5s, 1.0s, 1.5s, 2.0s, 2.5s, 3.0s.

1. **(0, 1)**
2. **(0, 2)**
3. **(0, 3)**
4. **(0, 4)**
5. **(0, 5)**
"""
        with self.assertRaises(ValueError):
            self.check_trajectory(response, [])


    def test_original_agentdriver_parser(self):
        """Ensure the original non-Qwen parser still works."""
        response = """
Planned Trajectory:
[(0, 1), (0, 2), (0, 3),
 (0, 4), (0, 5), (0, 6)]
"""

        with patch(
            "agentdriver.planning.motion_planning.generate_messages",
            return_value=("test_token", "test prompt", None)
        ), patch(
            "agentdriver.planning.motion_planning.run_one_round_conversation",
            return_value=([], {"content": response})
        ):
            traj, _ = planning_single_inference(
                planner_model_id="test-original-model",
                data_sample={},
                self_reflection=False,
                verbose=False,
            )

        np.testing.assert_allclose(
            traj,
            np.array([
                (0, 1), (0, 2), (0, 3),
                (0, 4), (0, 5), (0, 6)
            ], dtype=float),
        )

if __name__ == "__main__":
    unittest.main()
