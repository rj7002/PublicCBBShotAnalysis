import pandas as pd
import numpy as np

# TODO: store coordinates in csv file
class CourtCoordinates:
    def __init__(self, year):
        self.hoop_loc_x = 0
        self.hoop_loc_y = 52
        self.hoop_loc_z = 100
        self.court_perimeter_coordinates = []
        self.three_point_line_coordinates = []
        self.backboard_coordinates = []
        self.hoop_coordinates = []
        self.free_throw_line_coordinates = []
        self.court_lines_coordinates_df = pd.DataFrame()
        self.year = year

    @staticmethod

    def calculate_quadratic_values(a, b, c):
        '''
        Given values a, b, and c,
        the function returns the output of the quadratic formula
        '''
        x1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
        x2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

        return x1, x2

    def calculate_court_perimeter_coordinates(self):
        # half court lines
        # x goes from 250 to -250 (50 feet wide)
        # y should go from 0 to 470 (full court is 94 feet long, half court is 47 feet)
        court_perimeter_bounds = [[-250, 0, 0], [250, 0, 0], [250, 450, 0], [-250, 450, 0], [-250, 0, 0]]

        self.court_perimeter_coordinates = court_perimeter_bounds

    def calculate_three_point_line_coordinates(self):
        # Determine the radius for the 3-point line based on the year
        if int(self.year.split('-')[0]) < 2019:
            d = 205  # Use 220 for years before 1997
            d2 = 203
            
        else:
            d = 220  # Use 210 for years after 1997 (adjusted to 210 as per your requirement)
            d2 = 218
        # 3-point line left side coordinates
        line_coordinates = [[-d2, 0, 0], [-d2, 82, 0]]
        
        # 3-point line arc coefficients
        hoop_loc_x, hoop_loc_y = self.hoop_loc_x, self.hoop_loc_y
        a = 1
        b = -2 * 52

        # Iterate over x-coordinates for the 3-point arc
        for x_coord in range(-218, 218, 2):
            # Equation to find y-coordinate based on the circle's radius
            c = hoop_loc_y ** 2 + (hoop_loc_x - x_coord) ** 2 - (d) ** 2
            
            # Calculate the discriminant
            discriminant = b ** 2 - 4 * a * c
            
            # Check if the discriminant is non-negative (valid square root)
            if discriminant >= 0:
                y_coord = (-b + discriminant ** 0.5) / (2 * a)
                line_coordinates.append([x_coord, y_coord, 0])
            else:
                # If the discriminant is negative, skip this x-coordinate or handle it differently
                print(f"Skipping x={x_coord} due to negative discriminant")

        # 3-point line right side coordinates
        line_coordinates.append([d2, 82, 0])
        line_coordinates.append([d2, 0, 0])

        # Store the 3-point line coordinates
        self.three_point_line_coordinates = line_coordinates


    def calculate_backboard_coordinates(self):
        backboard_coordinates = [[30, 40, 95], [30, 40, 130], [-30, 40, 130], [-30, 40, 95], [30, 40, 95]]

        self.backboard_coordinates = backboard_coordinates
    
    def calculate_backboard_coordinates2(self):
        backboard_coordinates = [[10, 40, 100], [10, 40, 115], [-10, 40, 115], [-10, 40, 100], [10, 40, 100]]

        self.backboard_coordinates = backboard_coordinates

    def calculate_hoop_coordinates(self):
        hoop_coordinates_top_half = []
        hoop_coordinates_bottom_half = []

        hoop_center_x, hoop_center_y, hoop_center_z = (self.hoop_loc_x, self.hoop_loc_y, self.hoop_loc_z)
        hoop_min_x, hoop_max_x = (-7.5, 7.5)
        hoop_step = 0.5
        hoop_radius = 7.5

        a = 1
        b = -2 * hoop_center_y
        for hoop_coord_x in np.arange(hoop_min_x, hoop_max_x + hoop_step, hoop_step):
            c = hoop_center_y ** 2 + (hoop_center_x - hoop_coord_x) ** 2 - hoop_radius ** 2
            hoop_coord_y1, hoop_coord_y2 = self.calculate_quadratic_values(a, b, c)

            hoop_coordinates_top_half.append([hoop_coord_x, hoop_coord_y1, hoop_center_z])
            hoop_coordinates_bottom_half.append([hoop_coord_x, hoop_coord_y2, hoop_center_z])

        self.hoop_coordinates = hoop_coordinates_top_half + hoop_coordinates_bottom_half[::-1]
    
    def calculate_free_throw_line_coordinates(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        for i in range(num_points):
            angle = np.pi * i / (num_points - 1)  # Semi-circle from 0 to π
            x = circle_center[0] + radius * np.cos(angle)
            y = circle_center[1] + radius * np.sin(angle)
            circle_points.append([x, y, 0])
        
        # Adding lines to the free throw line
        free_throw_line_coordinates = [
            [-90, circle_center[1], 0],  # Left end of the line
            [90, circle_center[1], 0]    # Right end of the line
        ]
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline = [
                [-90, circle_center[1], 0],  # Left end of the free throw line to baseline
                [-90, baseline_y, 0],
                [90, circle_center[1], 0],   # Right end of the free throw line to baseline
                [90, baseline_y, 0]
            ]
        lines_to_baseline2 = [
                [-75, circle_center[1], 0],  # Left end of the free throw line to baseline
                [-75, baseline_y, 0],
                [75, circle_center[1], 0],   # Right end of the free throw line to baseline
                [75, baseline_y, 0]
            ]

    # Combine all coordinates
        self.free_throw_line_coordinates = circle_points + free_throw_line_coordinates 
    
    def calculate_free_throw_line_coordinates2(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline = [
                [-90, circle_center[1], 0],  # Left end of the free throw line to baseline
                [-90, baseline_y, 0]]
        self.free_throw_line_coordinates = lines_to_baseline
    
    def calculate_free_throw_line_coordinates3(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline = [
                [90, circle_center[1], 0],   # Right end of the free throw line to baseline
                [90, baseline_y, 0]
            ]
        self.free_throw_line_coordinates = lines_to_baseline

    def calculate_free_throw_line_coordinates4(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline2 = [
                [-75, circle_center[1], 0],  # Left end of the free throw line to baseline
                [-75, baseline_y, 0]
            ]
        self.free_throw_line_coordinates = lines_to_baseline2
    
    def calculate_free_throw_line_coordinates5(self):
        radius = 75  # 15 feet to inches
        distance_from_backboard = 15  # 15 feet to inches

        # Free throw line (semi-circle)
        circle_center = [0, 200 - 25, 0]  # Center of the semi-circle
        circle_points = []
        num_points = 100
        baseline_y = 0  # Baseline position (adjust as necessary)
        lines_to_baseline2 = [
                [75, circle_center[1], 0],  # Left end of the free throw line to baseline
                [75, baseline_y, 0]
            ]
        self.free_throw_line_coordinates = lines_to_baseline2
    def __get_hoop_coordinates2(self):
        num_net_lines = 10  # Number of vertical lines in the net
        net_length = 1.75 *10 # Length of the net hanging down from the hoop (in feet)
        initial_radius = 7.5  # Radius at the top of the net

        hoop_net_coordinates = []
        hoop_loc_x, hoop_loc_y, hoop_loc_z = self.hoop_loc_x, self.hoop_loc_y, self.hoop_loc_z


        for i in range(num_net_lines):
            angle = (i * 2 * np.pi) / num_net_lines
            
            for j in np.linspace(0, net_length, num=10):
                # Decrease the radius from the initial radius to half of it at the bottom
                current_radius = initial_radius * (1 - (j / net_length) * 0.5)
                
                x = hoop_loc_x + current_radius * np.cos(angle)
                y = hoop_loc_y + current_radius * np.sin(angle)
                z = hoop_loc_z - j
                
                hoop_net_coordinates.append([x, y, z])
        
        # Add lines on the other side (negative angles)
        for i in range(num_net_lines):
            angle = (i * 2 * np.pi) / num_net_lines + np.pi  # Shift angles to cover the opposite side
            
            for j in np.linspace(0, net_length, num=10):
                current_radius = initial_radius * (1 - (j / net_length) * 0.5)
                
                x = hoop_loc_x + current_radius * np.cos(angle)
                y = hoop_loc_y + current_radius * np.sin(angle)
                z = hoop_loc_z - j
                
                hoop_net_coordinates.append([x, y, z])
        self.hoop_net_coordinates = hoop_net_coordinates



    def calculate_court_lines_coordinates(self):
        self.calculate_court_perimeter_coordinates()
        court_df = pd.DataFrame(self.court_perimeter_coordinates, columns=['x', 'y', 'z'])
        court_df['line_id'] = 'outside_perimeter'
        court_df['line_group_id'] = 'court'

        self.calculate_three_point_line_coordinates()
        three_point_line_df = pd.DataFrame(self.three_point_line_coordinates, columns=['x', 'y', 'z'])
        three_point_line_df['line_id'] = 'three_point_line'
        three_point_line_df['line_group_id'] = 'court'

        self.calculate_backboard_coordinates()
        backboard_df = pd.DataFrame(self.backboard_coordinates, columns=['x', 'y', 'z'])
        backboard_df['line_id'] = 'backboard'
        backboard_df['line_group_id'] = 'backboard'

        self.calculate_backboard_coordinates2()
        backboard_df2 = pd.DataFrame(self.backboard_coordinates, columns=['x', 'y', 'z'])
        backboard_df2['line_id'] = 'backboard2'
        backboard_df2['line_group_id'] = 'backboard2'
            
        self.__get_hoop_coordinates2()
        netdf = pd.DataFrame(self.hoop_net_coordinates, columns=['x', 'y', 'z'])
        netdf['line_id'] = 'hoop2'
        netdf['line_group_id'] = 'hoop2'

        self.calculate_hoop_coordinates()
        hoop_df = pd.DataFrame(self.hoop_coordinates, columns=['x', 'y', 'z'])
        hoop_df['line_id'] = 'hoop'
        hoop_df['line_group_id'] = 'hoop'

        self.calculate_free_throw_line_coordinates()
        free_throw_line_df = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df['line_id'] = 'free_throw_line'
        free_throw_line_df['line_group_id'] = 'free_throw_line'

        self.calculate_free_throw_line_coordinates2()
        free_throw_line_df2 = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df2['line_id'] = 'free_throw_line2'
        free_throw_line_df2['line_group_id'] = 'free_throw_line2'

        self.calculate_free_throw_line_coordinates3()
        free_throw_line_df3 = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df3['line_id'] = 'free_throw_line3'
        free_throw_line_df3['line_group_id'] = 'free_throw_line3'

        self.calculate_free_throw_line_coordinates4()
        free_throw_line_df4 = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df4['line_id'] = 'free_throw_line4'
        free_throw_line_df4['line_group_id'] = 'free_throw_line4'

        self.calculate_free_throw_line_coordinates5()
        free_throw_line_df5 = pd.DataFrame(self.free_throw_line_coordinates, columns=['x', 'y', 'z'])
        free_throw_line_df5['line_id'] = 'free_throw_line5'
        free_throw_line_df5['line_group_id'] = 'free_throw_line5'


        self.court_lines_coordinates_df = pd.concat([
            court_df, three_point_line_df, backboard_df, hoop_df, free_throw_line_df,netdf,free_throw_line_df2,free_throw_line_df3,free_throw_line_df4,free_throw_line_df5,backboard_df2
        ], ignore_index=True, axis=0)

    def get_coordinates(self):
        self.calculate_court_lines_coordinates()
        return self.court_lines_coordinates_df
