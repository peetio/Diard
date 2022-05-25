var colors = ["#d1a3a3", "#d1cea3", "#a5d1a3", "#a3d1cc", "#a3b1d1", "#baa3d1", "#d1a3bf", "#d48585"]; // section colors
var color_n = 0; // section color index
var curr_section = 0;
var block;

/**
 * Creates new section in table of contents
 * @param  {String} text Heading content
 * @param  {String} color Section specific color  
 * @param  {Node} table Table container node
 * @param  {String} title_id #ID of sub title
 * @param  {Bool} has_sub If true section has sub sections (see hasSubSections(...))
 */
function addSectionToTOC(text, color, table, title_id, has_sub) {
    var container = document.createElement("div");
    var button = document.createElement("button");
    button.type = "button";
    button.classList.add("collapsible");

    var icon_p = document.createElement("p");
    var icon = document.createElement("span");
    icon.classList.add("material-icons");

    if (has_sub == true) {
        icon.innerHTML = "expand_more";
        icon_p.classList.add("rotate-90")
        icon_p.classList.add("icon_container")

    } else {
        icon.innerHTML = "fiber_manual_record";
        icon_p.classList.add("dot")
    }

    icon.style.color = color;
    icon_p.appendChild(icon);
    button.appendChild(icon_p);

    var link = document.createElement("a")
    link.href = '#' + title_id;
    link.innerHTML = text;

    var title = document.createElement("p");
    title.appendChild(link)
    button.appendChild(title);

    var hr = document.createElement("hr");
    button.appendChild(hr);
    container.appendChild(button);

    var sub_container = document.createElement("div");
    sub_container.classList.add("content");
    container.appendChild(sub_container);

    table.appendChild(container);
}

/**
 * Adds sub section elements to specified section in table of contents
 * @param  {String} text Subheading content
 * @param  {Node}   sub_section Content div to place sub sections
 * @param  {String} title_id #ID of sub title
 */
function addSubSectionToTOC(text, sub_section, title_id) {
    var sub_title = document.createElement("p");
    var link = document.createElement("a")

    link.href = '#' + title_id;
    link.innerHTML = text;
    sub_title.appendChild(link);
    sub_section.appendChild(sub_title);

    var hr = document.createElement("hr");
    sub_section.appendChild(hr);
}

/**
 * Recursive function to check if a section has sub sections (titles)
 * @param  {Number} initial_section Section number of to be checked section
 * @param  {Node}   node Sibling of the previous node
 */
function hasSubSections(initial_section, node) {
    // Check if next sibling is 'h2' tag
    var has_sub = false;
    var section = node.className.split(' ')[0];

    if (initial_section == section) {
        tag = node.tagName;
        if (tag == 'H2') {
            has_sub = true;
        }
    }

    // Check if next sibling belongs to same section
    sibling = node.nextElementSibling;
    if (has_sub || (initial_section !== section) || sibling == null) {
        return has_sub;
    }
    // No 'h2' tag found, check next sibling
    else {
        return hasSubSections(initial_section, sibling);
    }
}

window.addEventListener('load', function() {
    var layout = document.getElementById('layout');

    // Gets each layout element
    var layout_els = layout.getElementsByTagName('*');
    var table = document.body.getElementsByClassName('table')[0];

    var i;
    for (i = 0; i < layout_els.length; i++) {
        // Extract node attributes
        block = layout_els[i];
        section = block.className.split(' ')[0];
        tag = block.tagName;
        text = block.innerHTML;
        toc_items = table.getElementsByClassName('content');
        sub_section = toc_items[toc_items.length - 1];

        if (section) {
            // If new section
            if (section != curr_section) {
                curr_section = section;
                color_n++;
                if (color_n == colors.length) {
                    color_n = 0;
                }
                // Add section heading to table of contents
                if (tag == 'H2') {
                    sibling = block.nextElementSibling;
                    title_id = block.id;
                    has_sub = hasSubSections(section, sibling);
                    addSectionToTOC(text, colors[color_n], table, title_id, has_sub);
                }
            } else {
                // Add sub section to current section in table of contents
                if (tag == 'H2') {
                    if (typeof sub_section !== 'undefined') {
                        title_id = block.id;
                        addSubSectionToTOC(text, sub_section, title_id);
                    }
                }
            }

            // Nodes belonging to same section get same border color
            color = colors[color_n];
            block.style.borderLeft = "10px solid".concat(color);
        }
    };

    // Make each section in table of contents collapsable
    var colls = document.getElementsByClassName("collapsible");

    for (i = 0; i < colls.length; i++) {
        colls[i].addEventListener("click", function() {
            this.classList.toggle("active");
            var icon = this.getElementsByClassName("icon_container")[0];
            icon.classList.toggle("rotate-90")
            var content = this.nextElementSibling;

            if (content.style.display === "block") {
                content.style.display = "none";
            } else {
                content.style.display = "block";
            }
        });
    }
})
