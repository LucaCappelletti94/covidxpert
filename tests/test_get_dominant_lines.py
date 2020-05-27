from covidxpert.rotate.get_dominant_lines import get_dominant_lines

def test_get_dominant_lines():

    valid_line = [
        [
            [0.5, 0.5, 0, 1] # vertical line in the middle
        ]
    ]

    return valid_line[0] == list(get_dominant_lines(valid_line, 1, 1))

    line_on_side = [
        [
            [0, 1, 0, 1] # diagonal
        ]
    ]

    return not list(get_dominant_lines(line_on_side, 1, 1))
