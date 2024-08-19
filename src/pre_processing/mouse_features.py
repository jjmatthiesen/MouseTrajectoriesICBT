from src.utils.imports import *
from src.utils import utils as ut_utils


class Features(object):
    def __init__(self, df: pd.DataFrame, u_window, u_screen, u_document, u_id, p_id, a_id, iterration, file_save_name="features_all.csv"):
        """
        :param df: dataframe of the mouse data from one session
        :param u_window: size of user window e.g. [1090, 388]
        :param u_screen: size of user screen e.g. [1440, 900]
        :param u_document: size of document (page) e.g [1090, 1850]
        :param u_id: user id; used in db
        :param p_id: participant id; used in p2
        :param a_id: assessment id; e.g. 1939484: "DANA Screening"
        """
        self.df_mouse = df
        self.user_window = u_window
        self.user_screen = u_screen
        self.user_document = u_document
        self.user_id = u_id
        self.participant_id = p_id
        self.assessment_id = a_id
        self.i = iterration
        self.file_save_name = file_save_name

        self.session = np.array(
            self.df_mouse[
                ['eventtime', 'client_x', 'client_y', "page_x", "page_y", "eventname", "scrollspeed", "client_x_norm",
                 "client_y_norm"]])
        self.features = {'user_id': self.user_id, 'participant_id': self.participant_id,
                         'assessment_id': self.assessment_id}
        # use 'client' coordinated for feature calculation
        self.client_x_idx = 1
        self.client_y_idx = 2
        self.idx_move = np.where(self.session[:, 5] == "mousemove")[0]
        self.mouse_move = self.session[np.array(self.session[:, 5] == "mousemove")]

        # list of sub array for each movement sequence
        self.move_list = self.get_move_list()
        print(self.user_id)
        if len(self.mouse_move) >= 3:
            self.get_features()
            self.export_features()

    def export_features(self):
        """
        exports all created features to csv
        :return:
        """
        # mouse_tracker_id
        # patient_id
        # assessment_id
        pathlib.Path('../../data/features').mkdir(parents=True, exist_ok=True)
        with open('../../data/features/' + self.file_save_name, 'a', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(self.features.keys()))
            if self.i == 0:
                writer.writeheader()
            writer.writerow(self.features)

    def get_features(self):
        """
        creates all features
        :return:
        """
        x = np.array(self.session[self.idx_move, self.client_x_idx])
        y = np.array(self.session[self.idx_move, self.client_y_idx])

        # normalise using the users' screen
        x_norm = x / self.user_screen[0]
        y_norm = y / self.user_screen[1]

        # percentage of screen usage
        self.features['x_min_max_diff'] = (max(x) - min(x)) / self.user_screen[0]
        self.features['y_min_max_diff'] = (max(y) - min(y)) / self.user_screen[1]

        # angles
        angle_list = self.get_angle_list()
        self.features['angle_min'] = min(angle_list)
        self.features['angle_max'] = max(angle_list)
        self.features['angle_avg'] = np.mean(angle_list)

        # speed
        # in related work velocity is used
        # velocity is speed in a given direction. We do not mind the direction in mouse features
        speed_list = self.get_speed_list()
        speed_list[speed_list == 0] = np.nan
        self.features['speed_max'] = np.nanmax(speed_list)
        self.features['speed_min'] = np.nanmin(speed_list)
        self.features['speed_avg'] = np.nanmean(speed_list)

        # angle change rate
        angle_change_list = self.get_angle_list()
        # # if you moved ones straight, min is zero and therewith does not say much
        # # self.features['angle_change_min'] = np.min(np.abs(np.gradient(angle_list)))
        self.features['angle_change_max'] = np.nanmax(np.abs(np.gradient(angle_change_list)))
        self.features['angle_change_mean'] = np.nanmean(np.abs(np.gradient(angle_change_list)))

        # curvature
        curvature_list = self.get_curvature_list()
        # # we filter the list, since we want the focus on curves and not straight lines.
        # # 160 deg. is where we decided a curve-shape starts
        filtered_curvature_list = list(filter(lambda v: v < 160, curvature_list))
        # # we want to take the outliers (especially for small angels) into account
        # # and use therefore the average instead of the median.
        self.features['curvature_avg'] = np.mean(filtered_curvature_list)
        # # ut_plots.plot_trajectory_and_hist(filtered_curvature_list, movement, idx_move)

        # curvature change rate
        if len(curvature_list) > 1:
            self.features['curvature_change_mean'] = np.mean(np.abs(np.gradient(curvature_list)))
        else:
            self.features['curvature_change_mean'] = np.nan

        # amount of acute angle
        self.features['acute_angles'] = len(curvature_list[curvature_list < 90])

        # amount of obtuse angle
        self.features['obtuse_angles'] = len(curvature_list[curvature_list > 90])

        # horizontal speed
        horizontal_speed_list = np.array(self.get_horizontal_speed_list())
        # # during pauses the speed is 0. we replace those with nan for calculating the statistics
        horizontal_speed_list[horizontal_speed_list == 0] = np.nan
        self.features['horizontal_speed_min'] = np.nanmin(horizontal_speed_list)
        self.features['horizontal_speed_max'] = np.nanmax(horizontal_speed_list)
        self.features['horizontal_speed_avg'] = np.nanmean(horizontal_speed_list)

        # vertical speed
        vertical_speed_list = np.array(self.get_vertical_speed_list())
        # # during pauses the speed is 0. we replace those with nan for calculating the statistics
        vertical_speed_list[vertical_speed_list == 0] = np.nan
        self.features['vertical_speed_min'] = np.nanmin(vertical_speed_list)
        self.features['vertical_speed_max'] = np.nanmax(vertical_speed_list)
        self.features['vertical_speed_avg'] = np.nanmean(vertical_speed_list)

        # jitter
        # user must have moved more than 100px to determent a jitter
        if self.get_moved_dist() > 100:
            self.features['jitter'] = self.get_jitter_csaps(plot=False)
            self.features['jitter2'] = self.get_jitter_univariate_spline(plot=False)
        else:
            self.features['jitter'] = np.nan
            self.features['jitter2'] = np.nan

        # ----------additional features taken from (Feher et al.)-------
        # Duration of movement
        # # get time span of a movement until a pause, click, or scroll
        self.features['move_time_total'] = np.sum(np.array(self.get_movement_time_list()))

        # angular speed (Gamboa at al and Feher et al. uses angular velocity)
        time_diff_list = ut_utils.get_time_diff_list(self.mouse_move, 1, 2)
        angle_speed_list = angle_list / time_diff_list
        self.features['angle_speed_max'] = np.nanmax(np.abs(angle_speed_list))
        self.features['angle_speed_min'] = np.nanmin(np.abs(angle_speed_list))
        self.features['angle_speed_mean'] = np.nanmean(np.abs(angle_speed_list))

        # angular change speed (this is what Feher et al. and Gamboa et al. declare as angular velocity)
        # velocity is speed in a given direction. We take the absolut value her to be direction invariant.
        self.features['angle_change_speed_max'] = np.nanmax(np.abs(np.gradient(angle_list) / time_diff_list))
        self.features['angle_change_speed_min'] = np.nanmin(np.abs(np.gradient(angle_list) / time_diff_list))
        self.features['angle_change_speed_mean'] = np.nanmean(np.abs(np.gradient(angle_list) / time_diff_list))

        # ----------additional features from us ------------
        # Duration pauses
        self.features['pause_time_total'] = np.sum(np.array(self.get_pause_time_list()))

        # traveled dist
        self.features['moved_dist'] = self.get_moved_dist()
        # normalised traveled dist in percent
        self.features['moved_dist_norm'] = self.get_moved_dist_norm()

        # number of data points of the movement
        self.features['number_dp'] = len(self.session)

        # number of pauses
        self.features['pauses_no'] = len(np.where(self.session[:, 5] == "pause")[0])

        # number of clicks
        self.features['clicks_no'] = len(np.where(self.session[:, 5] == "mousedown")[0])

        # scroll speed
        scroll_list = self.session[self.session[:, 5] == 'scroll']
        # np.abs since scroll up give negative values
        if len(scroll_list) > 0:
            self.features['scroll_speed_min'] = np.nanmin(np.abs(scroll_list[:, 6]))
            self.features['scroll_speed_max'] = np.nanmean(np.abs(scroll_list[:, 6]))
            self.features['scroll_speed_mean'] = np.nanmean(np.abs(scroll_list[:, 6]))
        else:
            self.features['scroll_speed_min'] = 0.0
            self.features['scroll_speed_max'] = 0.0
            self.features['scroll_speed_mean'] = 0.0

        # dispersal
        x_min = min(self.session[:, self.client_x_idx])
        x_max = max(self.session[:, self.client_x_idx])
        y_min = min(self.session[:, self.client_y_idx])
        y_max = max(self.session[:, self.client_y_idx])
        dispersal_x = ut_utils.get_euc_dist(x_min, y_min, x_max, y_min)
        dispersal_y = ut_utils.get_euc_dist(x_min, y_min, x_min, y_max)
        self.features['dispersal_x_percent'] = dispersal_x / self.user_screen[0]
        self.features['dispersal_y_percent'] = dispersal_y / self.user_screen[1]
        self.features['area_percent'] = (dispersal_x * dispersal_y) / (self.user_screen[0] * self.user_screen[1])

    def get_move_list(self):
        """
        we need to just regard the mousemove events.
        Just filtering them would result in a big gab in time during a pause (not what we want)
        what we want:
        list of arrays (or dataframes; does not matter)
            example for id279
            move_list_sample = [movement[0:12], movement[129:137], movement[398:405]]
        """
        mov_list = []
        event_change_arr = np.array(self.df_mouse['event_change'])
        starts = np.where(event_change_arr == 1)[0]
        stops = np.where(event_change_arr == -1)[0]
        # if movement starts with 'mousemove' add 0 to begin of starts
        if self.session[0][5] == 'mousemove':
            starts = np.insert(starts, 0, 0)
        # if movement stops with 'mousemove' add len(movement) to end of stops
        if self.session[-1][5] == 'mousemove':
            stops = np.append(stops, len(self.session))

        for i, j in zip(starts, stops):
            if (j - i) > 1:
                mov_list.append(self.session[i: j])
        return mov_list

    def get_horizontal_speed_list(self):
        """
        speed in px/s
        :return:
        """
        hor_speed_list = []
        for mov in self.move_list:
            x_coords = mov[:, 1]
            time = mov[:, 0]
            time_diff = np.gradient(time) / 1000
            hor_speed_list.append(list(np.abs(np.gradient(x_coords) / time_diff)))
        return [item for sublist in hor_speed_list for item in sublist]

    def get_vertical_speed_list(self):
        """
        speed in px/s
        :return:
        """
        ver_speed_list = []
        for mov in self.move_list:
            y_coords = mov[:, 2]
            time = mov[:, 0]
            time_diff = np.gradient(time) / 1000
            ver_speed_list.append(list(np.abs(np.gradient(y_coords) / time_diff)))
        return [item for sublist in ver_speed_list for item in sublist]

    def get_angle_list(self):
        """
        List of angles of movement.
        We calculate the angles of movement between two points.
        :return:
        """
        angle_list = []
        for i in range(1, len(self.mouse_move)):
            x_coord = self.mouse_move[i][1]
            x_coord_prev = self.mouse_move[i - 1][1]
            y_coord = self.mouse_move[i][2]
            y_coord_prev = self.mouse_move[i - 1][2]
            angle = get_angle(x_coord_prev, y_coord_prev, x_coord, y_coord)
            angle_list.append(angle)
        return np.array(angle_list)

    def get_speed_list(self):
        """
        speed in px/s
        :return:
        """
        s_list = []
        for mov in self.move_list:
            for i in range(1, len(mov)):
                x_coord = mov[i][1]
                x_coord_prev = mov[i - 1][1]
                y_coord = mov[i][2]
                y_coord_prev = mov[i - 1][2]
                dist = ut_utils.get_euc_dist(x_coord_prev, y_coord_prev, x_coord, y_coord)
                time_diff = (mov[i][0] - mov[i - 1][0]) / 1000
                s_list.append(round((dist / time_diff), 2))
        return np.array(s_list)

    def get_curvature_list(self, plot_angles=False):
        """
        given three points a, b, c the curvature for is the smaller angle between these points.

        :param plot_angles: if the angle should be plotted
        :return:
        """
        curv_list = []
        for i in range(2, len(self.session)):
            x_1 = self.session[i - 2][1:3]
            x_2 = self.session[i - 1][1:3]
            x_3 = self.session[i][1:3]
            dist_1 = ut_utils.get_euc_dist(x_1[0], x_1[1], x_2[0], x_2[1])
            dist_2 = ut_utils.get_euc_dist(x_2[0], x_2[1], x_3[0], x_3[1])
            # if the movement between the point is just one pixel, we do not take it#s angle into account
            if dist_1 > 1.0 and dist_2 > 1.0:
                angle = get_angle_three(x_1, x_2, x_3)
                if plot_angles:
                    if angle < 120:
                        plt.plot([x_1[0], x_2[0], x_3[0]], [x_1[1], x_2[1], x_3[1]], c='#000000')
                        plt.title(str(angle) + " degree")
                        plt.show()
                curv_list.append(angle)
        return np.array(curv_list)

    def get_movement_time_list(self):
        """
        The times, where movement happened.
        :return:
        """
        movement_times = []
        move_start, move_stop = 0, 0

        for i, point in enumerate(self.session):
            if point[5] == 'mousemove':
                if move_start == 0:
                    move_start = point[0]
                move_stop = point[0]
            else:
                move_time_count = move_stop - move_start
                if move_time_count > 0:
                    movement_times.append(move_time_count)
                move_start, move_stop = 0, 0
        return movement_times

    def get_pause_time_list(self):
        """
        The times, where no movement happened (aka pauses).
        :return:
        """
        pause_times = []
        pause_start, pause_stop = 0, 0
        for i, point in enumerate(self.session):
            if point[5] == 'pause':
                if pause_start == 0:
                    pause_start = point[0]
                pause_stop = point[0]
            else:
                pause_time_count = pause_stop - pause_start
                if pause_time_count > 0:
                    pause_times.append(pause_time_count)
                pause_start, pause_stop = 0, 0
        return pause_times

    def get_moved_dist(self):
        """
        Calculates the total moved distance in pixels.
        :return:
        """
        moved_dist = 0
        for mov in self.move_list:
            for i in range(1, len(mov)):
                if mov[i][5] == 'mousemove' and mov[i - 1][5] == 'mousemove':
                    moved_dist += round(ut_utils.get_euc_dist(
                        mov[i - 1][1], mov[i - 1][2], mov[i][1], mov[i][2]), 0)
        return moved_dist

    def get_moved_dist_norm(self):
        """
        Calculates the total moved distance, normalised by the screen size in pixels.
        :return:
        """
        moved_dist = 0
        for move in self.move_list:
            for i in range(1, len(move) - 1):
                a = np.abs(move[i - 1][1] - move[i][1])
                a_norm = a / self.user_screen[0]
                b = np.abs(move[i - 1][2] - move[i][2])
                b_norm = b / self.user_screen[1]
                euc_dist_norm = math.sqrt(a_norm ** 2 + b_norm ** 2)
                moved_dist += euc_dist_norm
        return moved_dist

    def get_jitter_csaps(self, plot=False, save=False):
        """
        calculates the jitter using a Univariate Cubic Smoothing Spline.
        :param plot: If the trajectory should be plotted.
        :param save: If the plot should be saved.
        :return:
        """
        idx_move = np.where(self.session[:, 5] == "mousemove")[0]
        # using the client coordinated (screen, not page)
        x_coords = self.session[idx_move, 1]
        y_coords = self.session[idx_move, 2] * -1
        # equidistant points
        xd = np.diff(x_coords.astype(int))
        yd = np.diff(y_coords.astype(int))
        dist = np.sqrt(xd ** 2 + yd ** 2)
        u = np.cumsum(dist)
        u = np.hstack([[0], u])
        moved_dist = self.get_moved_dist()
        tn = np.linspace(0, u.max(), int(moved_dist / 25))
        # tn_norm = (tn-np.min(tn))/(np.max(tn)-np.min(tn))
        # tn = np.linspace(0, u.max(), 50)
        xn = np.interp(tn, u, x_coords.astype(int))
        yn = np.interp(tn, u, y_coords.astype(int))

        # csaps
        tn_sp = np.linspace(0, 1, len(xn))
        sp_x = csaps.UnivariateCubicSmoothingSpline(tn_sp, xn, smooth=1.0 - 1e-7)
        sp_y = csaps.UnivariateCubicSmoothingSpline(tn_sp, yn, smooth=1.0 - 1e-7)
        ts_n = np.linspace(tn_sp[0], tn_sp[-1], 150)
        xs = sp_x(tn_sp)
        ys = sp_y(tn_sp)

        # calculate the ratio between the original path length and the smoothed spline
        xs_diff = np.diff(xs.astype(int))
        ys_diff = np.diff(ys.astype(int))
        distance_eqd_points = sum(np.sqrt(xs_diff ** 2 + ys_diff ** 2))
        if plot:
            plt.plot(x_coords, y_coords, '-', label='data')
            plt.plot(xs, ys, '-', label='csaps')
            plt.title(str(1 - (distance_eqd_points / sum(dist))))
            plt.legend(loc='lower left')
            if save:
                plt.savefig("../../data/investigation/" + str(self.user_id) + "jitter_csaps.png")
                plt.savefig("../../data/investigation/" + str(self.user_id) + "jitter_csaps.svg")
                plt.close("all")
            else:
                plt.show()
        return 1 - (distance_eqd_points / sum(dist))

    def get_jitter_univariate_spline(self, plot=True, s=0.2, save=False):
        """
        Calculated the spline using the Univariate Spline.
        :param plot: If the trajectory should be plotted.
        :param s: Positive smoothing factor used to choose the number of knots.
        :param save: If the plot should be saved.
        :return:
        """
        idx_move = np.where(self.session[:, 5] == "mousemove")[0]
        # using the client coordinated (screen, not page)
        x_coords = self.session[idx_move, 1]
        y_coords = self.session[idx_move, 2] * -1
        # Another approach to jitter
        # using normalised coordinates and the univariate spline
        x_norm = (x_coords - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords))
        y_norm = (y_coords - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords))

        points = np.vstack((x_norm.astype(float), y_norm.astype(float))).T

        # Linear length along the line:
        distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
        distance = np.insert(distance, 0, 0) / distance[-1]

        # Build a list of the spline function, one for each dimension:
        splines = [UnivariateSpline(distance, coords, k=3, s=s) for coords in points.T]

        # Computed the spline for the asked distances:
        alpha = np.linspace(0, 1, 75)
        lists = [spl(alpha) for spl in splines]
        points_fitted = np.vstack(lists).T

        # calculate the ratio between the original path length and the smoothed spline
        # the points here are normalised
        points_x_diff = np.diff(points_fitted[:, 0])
        points_y_diff = np.diff(points_fitted[:, 1])
        distance_smooth_norm_points = sum(np.sqrt(points_x_diff ** 2 + points_y_diff ** 2))
        xd_norm = np.diff(x_norm.astype(np.float64))
        yd_norm = np.diff(y_norm.astype(np.float64))
        dist_norm = np.sqrt(xd_norm ** 2 + yd_norm ** 2)
        if plot:
            plt.plot(x_norm, y_norm, '-k', label='original points')
            plt.plot(*points_fitted.T, '-r', label='fitted spline k=3, s=' + str(s))
            plt.title(str(1 - (distance_smooth_norm_points / sum(dist_norm))))
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            if save:
                plt.savefig("../../data/investigation/" + str(self.user_id) + "_jitter_univariate.png")
                plt.savefig("../../data/investigation/" + str(self.user_id) + "_jitter_univariate.svg")
                plt.close("all")
            else:
                plt.show()
        return 1 - (distance_smooth_norm_points / sum(dist_norm))


def get_angle(x_coord_prev, y_coord_prev, x_coord, y_coord):
    """
    Calculates angles of movement between two points
    :param x_coord_prev: x coordinate of point one
    :param y_coord_prev: y coordinate of point one
    :param x_coord: x coordinate of point two
    :param y_coord: y coordinate of point two
    :return:
    """
    dx = x_coord - x_coord_prev
    dy = y_coord - y_coord_prev
    return math.degrees(math.atan2(dy, dx))


def get_angle_three(a, b, c):
    """
    Calculates angles of movement between three points.
    :param a: point one with [x,y] coordinates
    :param b: point two with [x,y] coordinates
    :param c: point three with [x,y] coordinates
    :return:
    """
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    # angles should be positive and always the smaller one (inside)
    ang = np.abs(np.abs(ang) - 360) if np.abs(ang) > 180 else np.abs(ang)
    return ang


if __name__ == '__main__':
    generate_features_by_file = False
    generate_features_by_user_all = True
    if generate_features_by_file:
        # folder_path = "../../data/mouse_data/participants_test/mouse/*"
        folder_path = "../../data/mouse_data/participants/desktop/*/*/*.csv"
        participants_sessions = glob(folder_path)
        i = 0
        for file in participants_sessions:
            path_screen_users = "../../data/mouse_data/raw/mousetracker.csv"

            df_mouse = pd.read_csv(file)
            df_mouse['page_y'] = df_mouse['page_y'] * -1
            df_users_screen = pd.read_csv(path_screen_users)

            user_id = file.split("id")[-1].split("_")[0]
            user_screen_info = df_users_screen.loc[(df_users_screen['id'] == int(user_id))]
            assessment_id = user_screen_info.iloc[0]['assessment_id']
            participant_id = user_screen_info.iloc[0]['participant_id']

            user_window, user_screen, user_document = ut_utils.get_view_size(user_screen_info.iloc[0], False)
            df_mouse['client_x_norm'] = np.array(df_mouse['client_x']) / user_screen[0]
            df_mouse['client_y_norm'] = np.array(df_mouse['client_y']) / user_screen[1]
            df_mouse['mousemove'] = (df_mouse['eventname'] == 'mousemove').astype(int)
            df_mouse['event_change'] = df_mouse['mousemove'].diff()
            session = np.array(
                df_mouse[
                    ['eventtime', 'client_x', 'client_y', "page_x", "page_y", "eventname", "scrollspeed", "client_x_norm",
                     "client_y_norm"]])
            if len(session) > 29:
                # if user is not a touch user, generate mouse features
                if not np.isin("touchstart", session[:, 5]):
                    Features(df=df_mouse, u_window=user_window, u_document=user_document, u_screen=user_screen,
                             u_id=user_id, p_id=participant_id, a_id=assessment_id, iterration=i)
                    i += 1
            else:
                pathlib.Path('../../data/preprocessed').mkdir(parents=True, exist_ok=True)
                with open('../../data/features/user_deselected.csv', 'a', encoding='utf-8') as f:
                    f.write(f'{user_id}, {participant_id},{assessment_id} \n')
    elif generate_features_by_user_all:
        folder_path = "../../data/mouse_data/participants/desktop/train_test/*/*"
        # folder_path = "../../data/mouse_data/participants/desktop/train/*/*"
        path_screen_users = "../../data/mouse_data/raw/mousetracker.csv"
        for i, user in enumerate(glob(folder_path + "/*/")):
            df_user = pd.DataFrame()
            user_id = 0
            for file in glob(user + "/*.csv"):
                file_data = pd.read_csv(file)
                df_user = pd.concat(([df_user, file_data]))
                user_id = file.split("id")[-1].split("_")[0]
            df_user['page_y'] = df_user['page_y'] * -1
            df_users_screen = pd.read_csv(path_screen_users)

            user_screen_info = df_users_screen.loc[(df_users_screen['id'] == int(user_id))]
            assessment_id = user_screen_info.iloc[0]['assessment_id']
            participant_id = user_screen_info.iloc[0]['participant_id']

            user_window, user_screen, user_document = ut_utils.get_view_size(user_screen_info.iloc[0], False)
            df_user['client_x_norm'] = np.array(df_user['client_x']) / user_screen[0]
            df_user['client_y_norm'] = np.array(df_user['client_y']) / user_screen[1]
            df_user['mousemove'] = (df_user['eventname'] == 'mousemove').astype(int)
            df_user['event_change'] = df_user['mousemove'].diff()
            session = np.array(
                df_user[
                    ['eventtime', 'client_x', 'client_y', "page_x", "page_y", "eventname", "scrollspeed",
                     "client_x_norm",
                     "client_y_norm"]])
            if len(session) > 29:
                # if user is not a touch user, generate mouse features
                if not np.isin("touchstart", session[:, 5]):
                    Features(df=df_user, u_window=user_window, u_document=user_document, u_screen=user_screen,
                             u_id=user_id, p_id=participant_id, a_id=assessment_id, iterration=i,
                             file_save_name="features_user_all.csv")
            else:
                pathlib.Path('../../data/preprocessed').mkdir(parents=True, exist_ok=True)
                with open('../../data/features/user_deselected_feature_user.csv', 'a', encoding='utf-8') as f:
                    f.write(f'{user_id}, {participant_id},{assessment_id} \n')
    else:
        print("no method selected")

