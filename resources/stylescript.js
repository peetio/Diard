var colors = ["#d1a3a3", "#d1cea3", "#a5d1a3", "#a3d1cc", "#a3b1d1", "#baa3d1", "#d1a3bf", "#d48585"]; /* section colors */
var color_n = 0
var curr_section = 0

window.addEventListener('load', function () {
  document.body.querySelectorAll('*').forEach(function(block) {
    /* read the class value for each block, this should be the section, compare them,
     *  change the color value until last color is used, then reset index to 0 */
    section = block.className 
    if (section) {
      if (section != curr_section) {
        curr_section = section;
        color_n++;
        if (color_n > colors.length) {
          color_n = 0
        }
      }
      block.style.borderLeft = "10px solid".concat(colors[color_n])
    }
  });

  const x = document.body.querySelectorAll('*')
  console.log(x)
})

