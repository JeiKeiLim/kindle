"""Module Description.

Repos to be considered:
    Python:
        - https://github.com/gaphor/gaphas (Hmm...)
            - https://gaphas.readthedocs.io/en/stable/guide.html

    Web:
        - https://github.com/awslabs/diagram-maker
        - https://github.com/jgraph/drawio-desktop  (Electron)
        - https://codepen.io/Ni55aN/pen/rpOKNb  (Javascript+Pug)

- Author: Jongkuk Lim
- Contact: lim.jeikei@gmail.com
"""

import math

import cairo
import gi

# fmt: off
gi.require_version("Gtk", "3.0")  # noqa: isort:skip
from gi.repository import Gtk  # noqa: isort:skip
# fmt: on

from gaphas import Canvas, GtkView
from gaphas.guide import GuidePainter
from gaphas.item import Line
from gaphas.painter import (
    BoundingBoxPainter,
    FreeHandPainter,
    HandlePainter,
    ItemPainter,
    PainterChain,
)
from gaphas.segment import LineSegmentPainter, Segment, segment_tool
from gaphas.tool import (
    hover_tool,
    item_tool,
    placement_tool,
    scroll_tool,
    view_focus_tool,
    zoom_tool,
)
from gaphas.tool.rubberband import RubberbandPainter, RubberbandState, rubberband_tool
from gaphas.util import text_extents, text_underline
from gaphas.util import path_ellipse, text_align, text_multiline
from gaphas.connector import Handle
from gaphas.item import NW, Element, Matrices


class Box(Element):
    """A Box has 4 handles (for a start):
    NW +---+ NE SW +---+ SE
    """

    def __init__(self, connections, width=10, height=10):
        super().__init__(connections, width, height)

    def draw(self, context):
        c = context.cairo
        nw = self._handles[NW].pos
        c.rectangle(nw.x, nw.y, self.width, self.height)
        if context.hovered:
            c.set_source_rgba(0.8, 0.8, 1, 0.8)
        else:
            c.set_source_rgba(1, 1, 1, 0.8)
        c.fill_preserve()
        c.set_source_rgb(0, 0, 0.8)
        c.stroke()


class Text(Matrices):
    """Simple item showing some text on the canvas."""

    def __init__(self, text=None, plain=False, multiline=False, align_x=1, align_y=-1):
        super().__init__()
        self.text = text is None and "Hello" or text
        self.plain = plain
        self.multiline = multiline
        self.align_x = align_x
        self.align_y = align_y

    def handles(self):
        return []

    def ports(self):
        return []

    def point(self, x, y):
        return 0

    def draw(self, context):
        cr = context.cairo
        if self.multiline:
            text_multiline(cr, 0, 0, self.text)
        elif self.plain:
            cr.show_text(self.text)
        else:
            text_align(cr, 0, 0, self.text, self.align_x, self.align_y)


class Circle(Matrices):
    def __init__(self):
        super().__init__()
        self._handles = [Handle(), Handle()]
        h1, h2 = self._handles
        h1.movable = False

    @property
    def radius(self):
        h1, h2 = self._handles
        p1, p2 = h1.pos, h2.pos
        return ((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2) ** 0.5

    @radius.setter
    def radius(self, r):
        h1, h2 = self._handles
        h2.pos.x = r
        h2.pos.y = r

    def handles(self):
        return self._handles

    def ports(self):
        return []

    def point(self, x, y):
        h1, _ = self._handles
        p1 = h1.pos
        dist = ((x - p1.x) ** 2 + (y - p1.y) ** 2) ** 0.5
        return dist - self.radius

    def draw(self, context):
        cr = context.cairo
        path_ellipse(cr, 0, 0, 2 * self.radius, 2 * self.radius)
        cr.stroke()


# Global undo list
undo_list = []


def undo_handler(event):
    global undo_list
    undo_list.append(event)


def factory(view, cls):
    """Simple canvas item factory."""

    def wrapper():
        item = cls(view.model.connections)
        view.model.add(item)
        return item

    return wrapper


class MyBox(Box):
    """Box with an example connection protocol."""


class MyLine(Line):
    """Line with experimental connection protocol."""

    def __init__(self, connections):
        super().__init__(connections)
        self.fuzziness = 2

    def draw_head(self, context):
        cr = context.cairo
        cr.move_to(0, 0)
        cr.line_to(10, 10)
        cr.stroke()
        # Start point for the line to the next handle
        cr.move_to(0, 0)

    def draw_tail(self, context):
        cr = context.cairo
        cr.line_to(0, 0)
        cr.line_to(10, 10)
        cr.stroke()


class MyText(Text):
    """Text with experimental connection protocol."""

    def draw(self, context):
        Text.draw(self, context)
        cr = context.cairo
        w, h = text_extents(cr, self.text, multiline=self.multiline)
        cr.rectangle(0, 0, w, h)
        cr.set_source_rgba(0.3, 0.3, 1.0, 0.6)
        cr.stroke()


class UnderlineText(Text):
    def draw(self, context):
        cr = context.cairo
        text_underline(cr, 0, 0, "Some text(y)")


def rubberband_state(view):
    try:
        return view.rubberband_state
    except AttributeError:
        view.rubberband_state = RubberbandState()
        return view.rubberband_state


def apply_default_tool_set(view):
    view.remove_all_controllers()
    view.add_controller(item_tool(view))
    view.add_controller(segment_tool(view))
    view.add_controller(scroll_tool(view))
    view.add_controller(zoom_tool(view))
    view.add_controller(view_focus_tool(view))

    view.add_controller(rubberband_tool(view, rubberband_state(view)))
    view.add_controller(hover_tool(view))
    return rubberband_state


def apply_placement_tool_set(view, item_type, handle_index):
    def unset_placement_tool(gesture, offset_x, offset_y):
        apply_default_tool_set(view)

    view.remove_all_controllers()
    tool = placement_tool(view, factory(view, item_type), handle_index)
    tool.connect("drag-end", unset_placement_tool)
    view.add_controller(scroll_tool(view))
    view.add_controller(zoom_tool(view))
    view.add_controller(view_focus_tool(view))
    view.add_controller(tool)


def apply_painters(view):
    painter = FreeHandPainter(ItemPainter(view.selection))
    view.painter = (
        PainterChain()
        .append(painter)
        .append(HandlePainter(view))
        .append(LineSegmentPainter(view.selection))
        .append(GuidePainter(view))
        .append(RubberbandPainter(rubberband_state(view)))
    )
    view.bounding_box_painter = painter


def create_window(canvas, title, zoom=1.0):  # noqa too complex
    view = GtkView()

    apply_default_tool_set(view)
    apply_painters(view)

    w = Gtk.Window()
    w.set_title(title)
    w.set_default_size(400, 120)
    h = Gtk.HBox()
    w.add(h)

    # VBox contains buttons that can be used to manipulate the canvas:
    v = Gtk.VBox()
    v.set_property("border-width", 3)
    v.set_property("spacing", 2)
    f = Gtk.Frame()
    f.set_property("border-width", 1)
    f.add(v)
    h.pack_start(f, False, True, 0)

    v.add(Gtk.Label.new("Item placement:"))

    b = Gtk.Button.new_with_label("Add box")

    def on_add_box_clicked(button):
        apply_placement_tool_set(view, MyBox, 2)

    b.connect("clicked", on_add_box_clicked)
    v.add(b)

    b = Gtk.Button.new_with_label("Add line")

    def on_add_line_clicked(button):
        apply_placement_tool_set(view, MyLine, -1)

    b.connect("clicked", on_add_line_clicked)
    v.add(b)

    v.add(Gtk.Label.new("Zooming:"))

    b = Gtk.Button.new_with_label("Zoom in")

    def on_zoom_in_clicked(button):
        view.zoom(1.2)

    b.connect("clicked", on_zoom_in_clicked)
    v.add(b)

    b = Gtk.Button.new_with_label("Zoom out")

    def on_zoom_out_clicked(button):
        view.zoom(1 / 1.2)

    b.connect("clicked", on_zoom_out_clicked)
    v.add(b)

    v.add(Gtk.Label.new("Misc:"))

    b = Gtk.Button.new_with_label("Split line")

    def on_split_line_clicked(button):
        selection = view.selection
        if isinstance(selection.focused_item, Line):
            segment = Segment(selection.focused_item, canvas)
            segment.split_segment(0)

    b.connect("clicked", on_split_line_clicked)
    v.add(b)

    b = Gtk.Button.new_with_label("Delete focused")

    def on_delete_focused_clicked(button):
        if view.selection.focused_item:
            canvas.remove(view.selection.focused_item)

    b.connect("clicked", on_delete_focused_clicked)
    v.add(b)

    v.add(Gtk.Label.new("Export:"))

    b = Gtk.Button.new_with_label("Write demo.png")

    def on_write_demo_png_clicked(button):
        assert view.model
        painter = ItemPainter()

        # Update bounding boxes with a temporary CairoContext
        # (used for stuff like calculating font metrics)
        tmpsurface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
        tmpcr = cairo.Context(tmpsurface)
        bounding_box = BoundingBoxPainter(painter).bounding_box(
            canvas.get_all_items(), tmpcr
        )
        tmpcr.show_page()
        tmpsurface.flush()

        surface = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, int(bounding_box.width), int(bounding_box.height)
        )
        cr = cairo.Context(surface)
        cr.translate(-bounding_box.x, -bounding_box.y)
        painter.paint(items=list(view.model.get_all_items()), cairo=cr)
        cr.show_page()
        surface.write_to_png("demo.png")

    b.connect("clicked", on_write_demo_png_clicked)
    v.add(b)

    b = Gtk.Button.new_with_label("Write demo.svg")

    def on_write_demo_svg_clicked(button):
        assert view.model
        painter = ItemPainter()

        # Update bounding boxes with a temporaly CairoContext
        # (used for stuff like calculating font metrics)
        tmpsurface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
        tmpcr = cairo.Context(tmpsurface)
        bounding_box = BoundingBoxPainter(painter).bounding_box(
            canvas.get_all_items(), tmpcr
        )
        tmpcr.show_page()
        tmpsurface.flush()

        surface = cairo.SVGSurface(
            "demo.svg", int(bounding_box.width), int(bounding_box.height)
        )
        cr = cairo.Context(surface)
        cr.translate(-bounding_box.x, -bounding_box.y)
        painter.paint(items=list(view.model.get_all_items()), cairo=cr)
        cr.show_page()
        surface.flush()
        surface.finish()

    b.connect("clicked", on_write_demo_svg_clicked)
    v.add(b)

    b = Gtk.Button.new_with_label("Dump QTree")

    def on_dump_qtree_clicked(button, li):
        view._qtree.dump()

    b.connect("clicked", on_dump_qtree_clicked, [0])
    v.add(b)

    b = Gtk.Button.new_with_label("Reattach (in place)")

    def on_reattach_clicked(button, li):
        view.model = None
        view.model = canvas

    b.connect("clicked", on_reattach_clicked, [0])
    v.add(b)

    # Add the actual View:

    view.model = canvas
    view.zoom(zoom)
    view.set_size_request(150, 120)
    s = Gtk.ScrolledWindow.new()
    s.set_policy(Gtk.PolicyType.ALWAYS, Gtk.PolicyType.ALWAYS)
    s.set_hexpand(True)
    s.add(view)
    h.add(s)

    w.show_all()

    w.connect("destroy", Gtk.main_quit)


def create_canvas(c=None):
    if not c:
        c = Canvas()
    b = MyBox(c.connections)
    b.min_width = 20
    b.min_height = 30
    b.matrix.translate(20, 20)
    b.width = b.height = 40
    c.add(b)

    bb = Box(c.connections)
    bb.matrix.translate(10, 10)
    c.add(bb, parent=b)

    bb = Box(c.connections)
    bb.matrix.rotate(math.pi / 1.567)
    c.add(bb, parent=b)

    circle = Circle()
    h1, h2 = circle.handles()
    circle.radius = 20
    circle.matrix.translate(50, 160)
    c.add(circle)

    pb = Box(c.connections, 60, 60)
    pb.min_width = 40
    pb.min_height = 50
    pb.matrix.translate(100, 20)
    c.add(pb)

    ut = UnderlineText()
    ut.matrix.translate(100, 130)
    c.add(ut)

    t = MyText("Single line")
    t.matrix.translate(100, 170)
    c.add(t)

    line = MyLine(c.connections)
    c.add(line)
    line.handles()[1].pos = (30, 30)
    segment = Segment(line, c)
    segment.split_segment(0, 3)
    line.matrix.translate(30, 80)
    line.orthogonal = True

    return c


def main():
    # State handling (a.k.a. undo handlers)

    def print_handler(event):
        print("event:", event)

    c = Canvas()

    create_window(c, "View created before")

    create_canvas(c)

    # Start the main application

    create_window(c, "View created after")

    Gtk.main()


if __name__ == "__main__":
    import sys

    if "-p" in sys.argv:
        print("Profiling...")
        import hotshot
        import hotshot.stats

        prof = hotshot.Profile("demo-gaphas.prof")
        prof.runcall(main)
        prof.close()
        stats = hotshot.stats.load("demo-gaphas.prof")
        stats.strip_dirs()
        stats.sort_stats("time", "calls")
        stats.print_stats(20)
    else:
        main()