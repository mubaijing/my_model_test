from motoropt.optimization.problem import DesignProblem


def test_design_problem_vector_to_dict():
    problem = DesignProblem(
        {
            "a": {"lower": 0.0, "upper": 1.0},
            "b": {"lower": 2.0, "upper": 3.0},
        }
    )
    assert problem.vector_to_dict([0.5, 2.5]) == {"a": 0.5, "b": 2.5}
