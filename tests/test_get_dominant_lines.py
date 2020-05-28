from covidxpert.rotate.get_dominant_lines import get_dominant_lines

def test_get_dominant_lines():

    valid_line = [
        [
            [0.5, 0, 0.5, 1] # vertical line in the middle
        ]
    ]

    assert tuple(*valid_line[0]) == next(get_dominant_lines(valid_line, 1, 1))

    line_on_side = [
        [
            [0, 0, 1, 1] # diagonal
        ]
    ]

    assert not list(get_dominant_lines(line_on_side, 1, 1))
