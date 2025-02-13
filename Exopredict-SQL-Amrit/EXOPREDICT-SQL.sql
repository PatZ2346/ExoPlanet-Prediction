-- Exported from QuickDBD: https://www.quickdatabasediagrams.com/
-- NOTE! If you have used non-SQL datatypes in your design, you will have to change these here.
-- upload tables in following order- "orbitalTemp", "clustering", "discovery"

CREATE TABLE "orbitalTemp" (
    "Planet_Name" VARCHAR(255)   NOT NULL,
    "Orbital_Eccentricity" FLOAT   NOT NULL,
    "Equilibrium_Temperature" INT   NOT NULL,
    "Orbit_Type" VARCHAR(100)   NOT NULL,
    "orb_id" int   NOT NULL,
    CONSTRAINT "pk_orbitalTemp" PRIMARY KEY (
        "orb_id"
     )
);

CREATE TABLE "clustering" (
    "Planet_Name" VARCHAR   NOT NULL,
    "Planet_Mass_Earth" FLOAT   NOT NULL,
    "Equilibrium_Temperature" INT   NOT NULL,
    "Prediction" INT   NOT NULL,
    "orb_id" int   NOT NULL,
    "cluster_id" int   NOT NULL,
    CONSTRAINT "pk_clustering" PRIMARY KEY (
        "cluster_id"
     )
);

CREATE TABLE "discovery" (
    "Discovery_Method" VARCHAR   NOT NULL,
    "Discovery_Facility" VARCHAR   NOT NULL,
    "Host_Star" VARCHAR   NOT NULL,
    "Discovery_Year" DATE   NOT NULL,
    "orb_id" int   NOT NULL,
    "disc_id" int   NOT NULL,
    CONSTRAINT "pk_discovery" PRIMARY KEY (
        "disc_id"
     )
);



ALTER TABLE "clustering" ADD CONSTRAINT "fk_clustering_orb_id" FOREIGN KEY("orb_id")
REFERENCES "orbitalTemp" ("orb_id");

ALTER TABLE "discovery" ADD CONSTRAINT "fk_discovery_orb_id" FOREIGN KEY("orb_id")
REFERENCES "orbitalTemp" ("orb_id");



