import Nav from "@/components/Nav";
import Hero from "@/components/Hero";
import ImageShowcase from "@/components/ImageShowcase";
import Pipeline from "@/components/Pipeline";
import Architecture from "@/components/Architecture";
import TechStack from "@/components/TechStack";
import Results from "@/components/Results";
import ApiShowcase from "@/components/ApiShowcase";
import DockerSection from "@/components/DockerSection";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <ImageShowcase />
        <Pipeline />
        <Architecture />
        <Results />
        <ApiShowcase />
        <TechStack />
        <DockerSection />
      </main>
      <Footer />
    </>
  );
}
