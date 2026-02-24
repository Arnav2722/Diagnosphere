import { Link } from "react-router-dom";
import {
  Activity,
  Mail,
  Phone,
  MapPin,
  Github,
  Twitter,
  Linkedin,
} from "lucide-react";

export function Footer() {
  return (
    <footer className="bg-card border-t border-border">
      <div className="container mx-auto px-4 py-12 md:py-16">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 lg:gap-12">
          {/* Brand */}
          <div className="space-y-4">
            <Link to="/" className="flex items-center gap-2">
              <div className="p-2 rounded-xl">
                {/* <div className="p-2 rounded-xl bg-gradient-to-br from-primary to-accent"> */}
                <img
                  src="/diagnosphere.png"
                  alt="Logo"
                  className="h-10 w-10 object-contain"
                />
              </div>
              <span className="font-display font-bold text-[25px] bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                Diagnosphere
              </span>
            </Link>
            <p className="text-muted-foreground text-sm max-w-xs">
              Medical diagnosis platform using advanced ML Models for accurate
              health predictions.
            </p>
            <div className="flex items-center gap-3 ml-24">
              <a
                href="https://github.com/Arnav2722/"
                className="p-2 rounded-lg bg-secondary hover:bg-primary/10 hover:text-primary transition-colors"
              >
                <Github className="h-4 w-4" />
              </a>
              <a
                href="https://github.com/jahnvi02-star"
                className="p-2 rounded-lg bg-secondary hover:bg-primary/10 hover:text-primary transition-colors"
              >
                <Github className="h-4 w-4" />
              </a>
            </div>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="font-display font-semibold text-foreground mb-4">
              Quick Links
            </h3>
            <ul className="space-y-3">
              {["Home", "Services", "About", "Contact"].map((item) => (
                <li key={item}>
                  <Link
                    to={item === "Home" ? "/" : `/${item.toLowerCase()}`}
                    className="text-muted-foreground hover:text-primary transition-colors text-sm"
                  >
                    {item}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Services */}
          <div>
            <h3 className="font-display font-semibold text-foreground mb-4">
              Diagnostics
            </h3>
            <ul className="space-y-3">
              {[
                "Diabetes",
                "Heart Disease",
                "Kidney Disease",
                "Liver Disease",
                "Breast Cancer",
                "Malaria",
                "Pneumonia",
              ].map((item) => (
                <li key={item}>
                  <Link
                    to={`/diagnosis/${item.toLowerCase().replace(" ", "-")}`}
                    className="text-muted-foreground hover:text-primary transition-colors text-sm"
                  >
                    {item} Detection
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Contact */}
          <div>
            <h3 className="font-display font-semibold text-foreground mb-4">
              Contact Us
            </h3>
            <ul className="space-y-3">
              <li className="flex items-center gap-3 text-muted-foreground text-sm">
                <Mail className="h-4 w-4 text-primary" />
                <span>info.diagnosphere@gmail.com</span>
              </li>
              <li className="flex items-center gap-3 text-muted-foreground text-sm">
                <Phone className="h-4 w-4 text-primary" />
                <span>+91 9829632587</span>
              </li>
              <li className="flex items-start gap-3 text-muted-foreground text-sm">
                <MapPin className="h-4 w-4 text-primary mt-0.5" />
                <span>New Delhi, India, 110049</span>
              </li>
            </ul>
          </div>
        </div>

        <div className="mt-12 pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-muted-foreground text-sm ">
            © {new Date().getFullYear()} Diagnosphere. All rights reserved.
          </p>
          {/* <div className="flex items-center gap-6">
            <Link
              to="/privacy"
              className="text-muted-foreground hover:text-primary transition-colors text-sm"
            >
              Privacy Policy
            </Link>
            <Link
              to="/terms"
              className="text-muted-foreground hover:text-primary transition-colors text-sm"
            >
              Terms of Service
            </Link>
          </div> */}
        </div>
      </div>
    </footer>
  );
}
