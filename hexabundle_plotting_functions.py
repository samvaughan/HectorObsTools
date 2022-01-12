import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.transforms import IdentityTransform, TransformedBbox, Bbox


class AngleAnnotation(Arc):
    """
    Draws an arc between two vectors which appears circular in display space.
    """
    def __init__(self, xy, p1, p2, size=75, unit="points", ax=None,
                 text="", textposition="inside", text_kw=None, **kwargs):
        """
        Parameters
        ----------
        xy, p1, p2 : tuple or array of two floats
            Center position and two points. Angle annotation is drawn between
            the two vectors connecting *p1* and *p2* with *xy*, respectively.
            Units are data coordinates.

        size : float
            Diameter of the angle annotation in units specified by *unit*.

        unit : str
            One of the following strings to specify the unit of *size*:

            * "pixels": pixels
            * "points": points, use points instead of pixels to not have a
              dependence on the DPI
            * "axes width", "axes height": relative units of Axes width, height
            * "axes min", "axes max": minimum or maximum of relative Axes
              width, height

        ax : `matplotlib.axes.Axes`
            The Axes to add the angle annotation to.

        text : str
            The text to mark the angle with.

        textposition : {"inside", "outside", "edge"}
            Whether to show the text in- or outside the arc. "edge" can be used
            for custom positions anchored at the arc's edge.

        text_kw : dict
            Dictionary of arguments passed to the Annotation.

        **kwargs
            Further parameters are passed to `matplotlib.patches.Arc`. Use this
            to specify, color, linewidth etc. of the arc.

        """
        self.ax = ax or plt.gca()
        self._xydata = xy  # in data coordinates
        self.vec1 = p1
        self.vec2 = p2
        self.size = size
        self.unit = unit
        self.textposition = textposition

        super().__init__(self._xydata, size, size, angle=0.0,
                         theta1=self.theta1, theta2=self.theta2, **kwargs)

        self.set_transform(IdentityTransform())
        self.ax.add_patch(self)

        self.kw = dict(ha="center", va="center",
                       xycoords=IdentityTransform(),
                       xytext=(0, 0), textcoords="offset points",
                       annotation_clip=True)
        self.kw.update(text_kw or {})
        self.text = ax.annotate(text, xy=self._center, **self.kw)

    def get_size(self):
        factor = 1.
        if self.unit == "points":
            factor = self.ax.figure.dpi / 72.
        elif self.unit[:4] == "axes":
            b = TransformedBbox(Bbox.from_bounds(0, 0, 1, 1),
                                self.ax.transAxes)
            dic = {"max": max(b.width, b.height),
                   "min": min(b.width, b.height),
                   "width": b.width, "height": b.height}
            factor = dic[self.unit[5:]]
        return self.size * factor

    def set_size(self, size):
        self.size = size

    def get_center_in_pixels(self):
        """return center in pixels"""
        return self.ax.transData.transform(self._xydata)

    def set_center(self, xy):
        """set center in data coordinates"""
        self._xydata = xy

    def get_theta(self, vec):
        vec_in_pixels = self.ax.transData.transform(vec) - self._center
        return np.rad2deg(np.arctan2(vec_in_pixels[1], vec_in_pixels[0]))

    def get_theta1(self):
        return self.get_theta(self.vec1)

    def get_theta2(self):
        return self.get_theta(self.vec2)

    def set_theta(self, angle):
        pass

    # Redefine attributes of the Arc to always give values in pixel space
    _center = property(get_center_in_pixels, set_center)
    theta1 = property(get_theta1, set_theta)
    theta2 = property(get_theta2, set_theta)
    width = property(get_size, set_size)
    height = property(get_size, set_size)

    # The following two methods are needed to update the text position.
    def draw(self, renderer):
        self.update_text()
        super().draw(renderer)

    def update_text(self):
        c = self._center
        s = self.get_size()
        angle_span = (self.theta2 - self.theta1) % 360
        angle = np.deg2rad(self.theta1 + angle_span / 2)
        r = s / 2
        if self.textposition == "inside":
            r = s / np.interp(angle_span, [60, 90, 135, 180],
                                          [3.3, 3.5, 3.8, 4])
        self.text.xy = c + r * np.array([np.cos(angle), np.sin(angle)])
        if self.textposition == "outside":
            def R90(a, r, w, h):
                if a < np.arctan(h/2/(r+w/2)):
                    return np.sqrt((r+w/2)**2 + (np.tan(a)*(r+w/2))**2)
                else:
                    c = np.sqrt((w/2)**2+(h/2)**2)
                    T = np.arcsin(c * np.cos(np.pi/2 - a + np.arcsin(h/2/c))/r)
                    xy = r * np.array([np.cos(a + T), np.sin(a + T)])
                    xy += np.array([w/2, h/2])
                    return np.sqrt(np.sum(xy**2))

            def R(a, r, w, h):
                aa = (a % (np.pi/4))*((a % (np.pi/2)) <= np.pi/4) + \
                     (np.pi/4 - (a % (np.pi/4)))*((a % (np.pi/2)) >= np.pi/4)
                return R90(aa, r, *[w, h][::int(np.sign(np.cos(2*a)))])

            bbox = self.text.get_window_extent()
            X = R(angle, r, bbox.width, bbox.height)
            trans = self.ax.figure.dpi_scale_trans.inverted()
            offs = trans.transform(((X-s/2), 0))[0] * 72
            self.text.set_position([offs*np.cos(angle), offs*np.sin(angle)])



def plot_field_xy(df, df_guides):

    plt.style.use('default')
    colour_dict = {"Sky-H7": "blue", "Sky-H3":"purple"}

    colours = ['blue' if "Sky-H7" in ID else 'red' if "Sky-H3" in ID else'black' for ID in df.ID]

    fig, ax = plt.subplots()
    ax.scatter(df.x, df.y, c=colours)
    ax.scatter(df_guides.x, df_guides.y, c='orange')
    ax.scatter(df.loc[df['type']==0, 'x'], df.loc[df['type'] ==0, 'y'], c='green')

    # ax.arrow(29, -21.25, 0, 0.25, width=0.025)

    # ax.arrow(29, -21.25, 0.25, 0, width=0.025)
    # ax.text(29, -21.0, 'N')
    # ax.text(29.25, -21.25, 'E')
    ax.set_aspect(1)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax.set_xlim(-300, 300)
    ax.set_ylim(-300, 300)

    return fig, ax


def plot_field(df, df_guides):

    plt.style.use('default')
    
    colours = ['blue' if "Sky-H7" in ID else 'red' if "Sky-H3" in ID else'black' for ID in df.ID]


    fig, ax = plt.subplots()
    ax.scatter(df.RA, df.DEC, c=colours)
    ax.scatter(df_guides.RA, df_guides.DEC, c='orange')
    ax.scatter(df.loc[df['type']==0, 'RA'], df.loc[df['type'] ==0, 'DEC'], c='green')

    ax.arrow(29, -21.25, 0, 0.25, width=0.025)

    ax.arrow(29, -21.25, 0.25, 0, width=0.025)
    ax.text(29, -21.0, 'N')
    ax.text(29.25, -21.25, 'E')
    ax.set_aspect(1)

    ax.set_xlabel('RA')
    ax.set_ylabel('DEC')

    return fig, ax


def plot_guide_rotations(df):

    plt.style.use('default')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(df.x, df.y, c='r', zorder=10)
    length = 10


    for index, row in df.iterrows():

        if not 'Sky' in str(row.ID):
            angle_degrees = (row.angs + np.pi) * 180/np.pi

            if angle_degrees>360:
                angle_degrees -= 360

            line_angle_eq_0 = [(row.x, row.y), (row.x + 7.5, row.y)]
            line_guide_tail = [(row.x, row.y), (row.x + length * np.cos(row.angs + np.pi), row.y + length * np.sin(row.angs + np.pi))]
            ax.annotate(f'{row.Hexabundle}', xy=(row.x, row.y), xytext=(-20, -20), textcoords='offset points', size=12, fontweight='bold')
            ax.plot(*zip(*line_guide_tail), c='k', linewidth=2)
            ax.plot(*zip(*line_angle_eq_0), c='k', linestyle='dashed', linewidth=0.5)
            am = AngleAnnotation((row.x, row.y), line_angle_eq_0[1], line_guide_tail[1], ax=ax, size=75, text=rf"$\alpha={angle_degrees:.1f}$")


    arrow_x_centre = 0.8
    arrow_y_centre = 0.95
    arrow_length = 0.1
    ax.arrow(x=arrow_x_centre, y=arrow_y_centre, dx=0.0, dy=-1 * arrow_length, transform=ax.transAxes, width=0.005, facecolor='k')
    ax.arrow(x=arrow_x_centre, y=arrow_y_centre, dx=arrow_length, dy=0.0, transform=ax.transAxes, width=0.005, facecolor='k')

    #N arrow
    ax.annotate('N', xy=(arrow_x_centre, arrow_y_centre), xytext=(arrow_x_centre, arrow_y_centre - 0.03 - arrow_length), xycoords=ax.transAxes,
            fontsize=16, fontweight='bold', va='top', ha='center')
    ax.annotate('E', xy=(arrow_x_centre, arrow_y_centre), xytext=(arrow_x_centre + arrow_length + 0.03, arrow_y_centre), xycoords=ax.transAxes,
            fontsize=16, fontweight='bold', va='center', ha='left')
    return fig, ax