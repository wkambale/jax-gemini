(function () {
  "use strict";

  function ensureScrollTopButton() {
    if (document.getElementById("jf-scroll-top")) {
      return;
    }

    var button = document.createElement("button");
    button.id = "jf-scroll-top";
    button.type = "button";
    button.className = "jf-scroll-top";
    button.setAttribute("aria-label", "Go to top");
    button.innerHTML =
      '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="m5 12 7-7 7 7"/></svg>';

    button.addEventListener("click", function () {
      var reducedMotion =
        window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      window.scrollTo({
        top: 0,
        left: 0,
        behavior: reducedMotion ? "instant" : "smooth",
      });
    });

    document.body.appendChild(button);

    var toggleVisibility = function () {
      if (window.scrollY > 420) {
        button.classList.add("is-visible");
      } else {
        button.classList.remove("is-visible");
      }
    };

    window.addEventListener("scroll", toggleVisibility, { passive: true });
    toggleVisibility();
  }

  function resetScrollOnRouteChange() {
    if (!window.location.hash) {
      window.scrollTo({ top: 0, left: 0, behavior: "instant" });
    }
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(function () {
      ensureScrollTopButton();
      resetScrollOnRouteChange();
    });
  } else {
    document.addEventListener("DOMContentLoaded", function () {
      ensureScrollTopButton();
    });
  }
})();
