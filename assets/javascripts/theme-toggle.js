(function () {
  var storageKey = 'theme_preference';
  var root = document.documentElement;
  var group = document.getElementById('theme-switch');
  if (!group) return;

  var buttons = Array.prototype.slice.call(group.querySelectorAll('[data-theme-mode]'));
  var systemDark = window.matchMedia('(prefers-color-scheme: dark)');
  var reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  var busy = false;

  function readMode() {
    var saved = null;
    try { saved = localStorage.getItem(storageKey); } catch (e) {}
    return saved === 'light' || saved === 'dark' ? saved : 'auto';
  }

  function saveMode(mode) {
    try {
      if (mode === 'auto') localStorage.removeItem(storageKey);
      else localStorage.setItem(storageKey, mode);
    } catch (e) {}
  }

  function themeFor(mode) {
    if (mode === 'auto') return systemDark.matches ? 'dark' : 'light';
    return mode;
  }

  function markButtons(mode) {
    buttons.forEach(function (b) {
      b.setAttribute('aria-pressed', String(b.getAttribute('data-theme-mode') === mode));
    });
  }

  function applyTheme(theme) {
    root.classList.toggle('dark-mode', theme === 'dark');
    root.style.colorScheme = theme;
  }

  function switchTheme(theme) {
    var current = root.classList.contains('dark-mode') ? 'dark' : 'light';
    if (theme === current) return;
    if (busy || !document.startViewTransition || reduceMotion.matches) {
      applyTheme(theme);
      return;
    }
    busy = true;
    root.classList.add('theme-transitioning');
    var transition = document.startViewTransition(function () { applyTheme(theme); });
    transition.ready.then(function () {
      return root.animate(
        { opacity: [0, 1] },
        { duration: 420, easing: 'ease', pseudoElement: '::view-transition-new(root)' }
      ).finished;
    }).catch(function () {});
    transition.finished.finally(function () {
      root.classList.remove('theme-transitioning');
      busy = false;
    });
  }

  markButtons(readMode());

  buttons.forEach(function (b) {
    b.addEventListener('click', function () {
      var mode = b.getAttribute('data-theme-mode');
      saveMode(mode);
      markButtons(mode);
      switchTheme(themeFor(mode));
    });
  });

  var onSystemChange = function () {
    if (readMode() === 'auto') switchTheme(themeFor('auto'));
  };
  if (systemDark.addEventListener) systemDark.addEventListener('change', onSystemChange);
  else if (systemDark.addListener) systemDark.addListener(onSystemChange);
}());
