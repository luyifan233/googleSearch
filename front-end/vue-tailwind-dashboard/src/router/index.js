import { createRouter, createWebHistory } from "vue-router";

import DashboardPage from "@/pages/master/DashboardPage.vue";

import GroupPage from "@/pages/GroupPage.vue";
import UserPage from "@/pages/UserPage.vue";
import AccountPage from "@/pages/AccountPage.vue";
import SessionPage from "@/pages/SessionPage.vue";
import FriendPage from "@/pages/FriendPage.vue";
import TransactionPage from "@/pages/TransactionPage.vue";

const routes = [
    {
        path: "/",
        name: "Dashboard",
        component: DashboardPage,
        children: [
            {
                path: "/user",
                name: "User",
                component: UserPage,
            },
            {
                path: "/group",
                name: "Group",
                component: GroupPage,
            },
            {
                path: "/account",
                name: "Account",
                component: AccountPage,
            },
            {
                path: "/session",
                name: "Session",
                component: SessionPage,
            },
            {
                path: "/friend",
                name: "Friend",
                component: FriendPage,
            },
            {
                path: "/transaction",
                name: "Transaction",
                component: TransactionPage,
            },
        ]
    },
];

const router = Router();
export default router;
function Router() {
    const router = createRouter({
        history: createWebHistory(),
        routes,
    });
    return router;
}