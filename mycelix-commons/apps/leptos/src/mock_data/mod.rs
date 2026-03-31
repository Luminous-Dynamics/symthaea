// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use commons_leptos_types::*;

fn now() -> i64 { (js_sys::Date::now() * 1000.0) as i64 }
fn hours_ago(h: i64) -> i64 { now() - h * 3_600_000_000 }
fn days_ago(d: i64) -> i64 { now() - d * 86_400_000_000 }
fn days_from_now(d: i64) -> i64 { now() + d * 86_400_000_000 }

pub fn mock_needs() -> Vec<NeedView> {
    vec![
        NeedView { hash: "need_001".into(), id: "need-001".into(), title: "Help moving furniture this weekend".into(), description: "Downsizing — need 2-3 people to help carry furniture on Saturday morning.".into(), category: NeedCategory::Housing, requester_did: "did:mycelix:kai".into(), urgency: Urgency::Medium, status: NeedStatus::Open, created: hours_ago(6) },
        NeedView { hash: "need_002".into(), id: "need-002".into(), title: "Emergency childcare needed tomorrow".into(), description: "Unexpected work shift — need someone to watch 2 kids (ages 4 and 7) from 9am-3pm.".into(), category: NeedCategory::Childcare, requester_did: "did:mycelix:luna".into(), urgency: Urgency::High, status: NeedStatus::Open, created: hours_ago(2) },
        NeedView { hash: "need_003".into(), id: "need-003".into(), title: "Ride to medical appointment".into(), description: "Need transport to clinic 15km away, Tuesday 10am. Can share fuel cost.".into(), category: NeedCategory::Transportation, requester_did: "did:mycelix:elder-oak".into(), urgency: Urgency::Medium, status: NeedStatus::Open, created: days_ago(1) },
    ]
}

pub fn mock_offers() -> Vec<OfferView> {
    vec![
        OfferView { hash: "offer_001".into(), id: "offer-001".into(), title: "Free sourdough starter".into(), description: "Have plenty of mature starter to share. Can deliver within 5km.".into(), category: NeedCategory::Food, offerer_did: "did:mycelix:aria".into(), created: days_ago(2) },
        OfferView { hash: "offer_002".into(), id: "offer-002".into(), title: "Tutoring in basic math and reading".into(), description: "Retired teacher, happy to help children or adults. Evenings and weekends.".into(), category: NeedCategory::Skills, offerer_did: "did:mycelix:sol".into(), created: days_ago(3) },
    ]
}

pub fn mock_care_circles() -> Vec<CareCircleView> {
    vec![
        CareCircleView { hash: "circle_001".into(), name: "Riverside Neighborhood Care".into(), description: "Neighbors helping neighbors — meal trains, rides, childcare swaps.".into(), circle_type: CircleType::Neighborhood, member_count: 23, active: true, created: days_ago(180) },
        CareCircleView { hash: "circle_002".into(), name: "Oak Street Elders Circle".into(), description: "Supporting our elders with daily tasks, companionship, and dignity.".into(), circle_type: CircleType::Family, member_count: 8, active: true, created: days_ago(90) },
    ]
}

pub fn mock_plots() -> Vec<PlotView> {
    vec![
        PlotView { hash: "plot_001".into(), name: "Riverside Community Garden".into(), area_sqm: 450.0, plot_type: PlotType::CommunityGarden, steward_did: "did:mycelix:river".into(), crop_count: 12 },
        PlotView { hash: "plot_002".into(), name: "Rooftop Herbs — Block C".into(), area_sqm: 30.0, plot_type: PlotType::Rooftop, steward_did: "did:mycelix:aria".into(), crop_count: 8 },
    ]
}

pub fn mock_markets() -> Vec<MarketView> {
    vec![
        MarketView { hash: "market_001".into(), name: "Saturday Morning Market".into(), market_type: MarketType::Farmers, listing_count: 14 },
        MarketView { hash: "market_002".into(), name: "Community Food Bank".into(), market_type: MarketType::FoodBank, listing_count: 6 },
    ]
}

pub fn mock_food_listings() -> Vec<FoodListingView> {
    vec![
        FoodListingView { hash: "food_001".into(), product_name: "Mixed salad greens".into(), quantity_kg: 2.5, price_per_kg: 0.0, organic: true, producer_did: "did:mycelix:river".into(), available: true },
        FoodListingView { hash: "food_002".into(), product_name: "Sourdough bread loaves".into(), quantity_kg: 5.0, price_per_kg: 0.5, organic: false, producer_did: "did:mycelix:aria".into(), available: true },
    ]
}

pub fn mock_water_systems() -> Vec<WaterSystemView> {
    vec![
        WaterSystemView { hash: "water_001".into(), name: "Community Hall Rainwater".into(), system_type: WaterSystemType::RoofRainwater, capacity_liters: 5000, current_level_liters: 3200, owner_did: "did:mycelix:commons".into() },
        WaterSystemView { hash: "water_002".into(), name: "Park Ground Catchment".into(), system_type: WaterSystemType::GroundCatchment, capacity_liters: 12000, current_level_liters: 8500, owner_did: "did:mycelix:commons".into() },
    ]
}

pub fn mock_tools() -> Vec<ToolView> {
    vec![
        ToolView { hash: "tool_001".into(), name: "Electric drill".into(), description: "Bosch 18V cordless drill with 2 batteries and bit set.".into(), category: ToolCategory::PowerTool, condition: ToolCondition::Good, available: true, owner_did: "did:mycelix:kai".into() },
        ToolView { hash: "tool_002".into(), name: "Wheelbarrow".into(), description: "Standard garden wheelbarrow, slightly rusty but functional.".into(), category: ToolCategory::Garden, condition: ToolCondition::Fair, available: true, owner_did: "did:mycelix:river".into() },
        ToolView { hash: "tool_003".into(), name: "Sewing machine".into(), description: "Singer mechanical sewing machine with manual.".into(), category: ToolCategory::Other("Textile".into()), condition: ToolCondition::Good, available: false, owner_did: "did:mycelix:luna".into() },
    ]
}

pub fn mock_events() -> Vec<EventView> {
    vec![
        EventView { hash: "event_001".into(), title: "Community garden planting day".into(), description: "Spring planting — bring gloves and seeds if you have them.".into(), category: "Garden".into(), organizer_did: "did:mycelix:river".into(), start_time: days_from_now(3), end_time: days_from_now(3) + 14_400_000_000, max_attendees: 30, rsvp_count: 12 },
        EventView { hash: "event_002".into(), title: "Tool library inventory check".into(), description: "Monthly tool audit and maintenance session.".into(), category: "Maintenance".into(), organizer_did: "did:mycelix:kai".into(), start_time: days_from_now(7), end_time: days_from_now(7) + 7_200_000_000, max_attendees: 10, rsvp_count: 4 },
        EventView { hash: "event_003".into(), title: "Care circle potluck".into(), description: "Monthly gathering for the Riverside neighborhood care circle.".into(), category: "Social".into(), organizer_did: "did:mycelix:aria".into(), start_time: days_from_now(5), end_time: days_from_now(5) + 10_800_000_000, max_attendees: 40, rsvp_count: 18 },
    ]
}
